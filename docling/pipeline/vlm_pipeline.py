import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Union, cast

from docling_core.types.doc import (
    BoundingBox,
    DocItem,
    DoclingDocument,
    ImageRef,
    PictureItem,
    ProvenanceItem,
    TextItem,
)
from docling_core.types.doc.base import (
    BoundingBox,
    Size,
)
from docling_core.types.doc.document import DocTagsDocument
from PIL import Image as PILImage

from docling.backend.abstract_backend import AbstractDocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.pdf_backend import PdfDocumentBackend
from docling.datamodel.base_models import InputFormat, Page
from docling.datamodel.document import ConversionResult, InputDocument
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    ApiVlmOptions,
    InferenceFramework,
    InlineVlmOptions,
    ResponseFormat,
)
from docling.datamodel.settings import settings
from docling.models.api_vlm_model import ApiVlmModel
from docling.models.vlm_models_inline.hf_transformers_model import (
    HuggingFaceTransformersVlmModel,
)
from docling.models.vlm_models_inline.mlx_model import HuggingFaceMlxModel
from docling.pipeline.base_pipeline import PaginatedPipeline
from docling.utils.profiling import ProfilingScope, TimeRecorder

_log = logging.getLogger(__name__)


class VlmPipeline(PaginatedPipeline):
    def __init__(self, pipeline_options: VlmPipelineOptions):
        super().__init__(pipeline_options)
        self.keep_backend = True

        self.pipeline_options: VlmPipelineOptions

        artifacts_path = None
        if pipeline_options.artifacts_path is not None:
            artifacts_path = Path(pipeline_options.artifacts_path).expanduser()
        elif settings.artifacts_path is not None:
            artifacts_path = Path(settings.artifacts_path).expanduser()

        if artifacts_path is not None and not artifacts_path.is_dir():
            raise RuntimeError(
                f"The value of {artifacts_path=} is not valid. "
                "When defined, it must point to a folder containing all models required by the pipeline."
            )

        self.force_backend_text = (
            pipeline_options.force_backend_text
            and pipeline_options.vlm_options.response_format == ResponseFormat.DOCTAGS
        )

        self.keep_images = self.pipeline_options.generate_page_images

        vlm_opts = pipeline_options.vlm_options
        if isinstance(vlm_opts, ApiVlmOptions):
            self.build_pipe = [ApiVlmModel(
                enabled=True,
                enable_remote_services=self.pipeline_options.enable_remote_services,
                vlm_options=cast(ApiVlmOptions, vlm_opts),
            )]
        elif isinstance(vlm_opts, InlineVlmOptions):
            inline_opts = cast(InlineVlmOptions, vlm_opts)
            if inline_opts.inference_framework == InferenceFramework.MLX:
                self.build_pipe = [HuggingFaceMlxModel(
                    enabled=True,
                    artifacts_path=artifacts_path,
                    accelerator_options=pipeline_options.accelerator_options,
                    vlm_options=inline_opts,
                )]
            elif inline_opts.inference_framework == InferenceFramework.TRANSFORMERS:
                self.build_pipe = [HuggingFaceTransformersVlmModel(
                    enabled=True,
                    artifacts_path=artifacts_path,
                    accelerator_options=pipeline_options.accelerator_options,
                    vlm_options=inline_opts,
                )]
            else:
                raise ValueError(
                    f"Could not instantiate the right type of VLM pipeline: {inline_opts.inference_framework}"
                )

        self.enrichment_pipe = []

    def initialize_page(self, conv_res: ConversionResult, page: Page) -> Page:
        with TimeRecorder(conv_res, "page_init"):
            page._backend = conv_res.input._backend.load_page(page.page_no)  # type: ignore
            if page._backend is not None and page._backend.is_valid():
                page.size = page._backend.get_size()
                page.parsed_page = page._backend.get_segmented_page()

        return page

    def extract_text_from_backend(
        self, page: Page, bbox: Union[BoundingBox, None]
    ) -> str:
        # Convert bounding box normalized to 0-100 into page coordinates for cropping
        return (
            page._backend.get_text_in_rect(bbox)
            if bbox and page.size and getattr(page, "_backend", None)
            else ""
        )

    def _assemble_document(self, conv_res: ConversionResult) -> ConversionResult:
        with TimeRecorder(conv_res, "doc_assemble", scope=ProfilingScope.DOCUMENT):
            if (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.DOCTAGS
            ):
                conv_res.document = self._turn_dt_into_doc(conv_res)

            elif (
                self.pipeline_options.vlm_options.response_format
                == ResponseFormat.MARKDOWN
            ):
                conv_res.document = self._turn_md_into_doc(conv_res)

            elif (
                self.pipeline_options.vlm_options.response_format == ResponseFormat.HTML
            ):
                conv_res.document = self._turn_html_into_doc(conv_res)

            else:
                raise RuntimeError(
                    f"Unsupported VLM response format {self.pipeline_options.vlm_options.response_format}"
                )

            # Generate images of the requested element types
            if self.pipeline_options.generate_picture_images:
                scale = self.pipeline_options.images_scale
                for element, _level in conv_res.document.iterate_items():
                    if not isinstance(element, DocItem) or len(element.prov) == 0:
                        continue
                    if (
                        isinstance(element, PictureItem)
                        and self.pipeline_options.generate_picture_images
                    ):
                        page_ix = element.prov[0].page_no - 1
                        page = conv_res.pages[page_ix]
                        assert page.size is not None
                        assert page.image is not None

                        crop_bbox = (
                            element.prov[0]
                            .bbox.scaled(scale=scale)
                            .to_top_left_origin(page_height=page.size.height * scale)
                        )

                        cropped_im = page.image.crop(crop_bbox.as_tuple())
                        element.image = ImageRef.from_pil(
                            cropped_im, dpi=int(72 * scale)
                        )

        return conv_res

    def _turn_dt_into_doc(self, conv_res) -> DoclingDocument:
        pages = conv_res.pages
        doctags_list = []
        image_list = []

        # Only allocate a dummy PIL image if we must
        dummy_image = None
        for page in pages:
            # Using fast local variables and branch order to reduce unnecessary allocations!
            predicted_doctags = ""
            img = None
            pr = getattr(page, 'predictions', None)
            if pr is not None:
                vlm_resp = getattr(pr, 'vlm_response', None)
                if vlm_resp is not None:
                    predicted_doctags = vlm_resp.text
            img = getattr(page, 'image', None)
            if img is None:
                if dummy_image is None:
                    dummy_image = PILImage.new("RGB", (1, 1), (255, 255, 255))
                img = dummy_image
            doctags_list.append(predicted_doctags)
            image_list.append(img)

        # No need to cast; type checkers only
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            doctags_list, image_list
        )
        conv_res.document = DoclingDocument.load_from_doctags(
            doctag_document=doctags_doc
        )

        # Backend text replacement, if and only if needed (do this in one pass after doc conversion for efficiency)
        if getattr(pages[0], 'size', None):  # Assume all pages have size if first does.
            if self.force_backend_text:
                scale = self.pipeline_options.images_scale
                # It is most efficient if we actually enumerate which page and element to use--if possible you can supply mapping here.
                page0 = pages[0]
                page_height = page0.size.height * scale
                for element, _ in conv_res.document.iterate_items():
                    if not (isinstance(element, TextItem) and element.prov):
                        continue
                    bbox = element.prov[0].bbox.scaled(scale=scale).to_top_left_origin(page_height=page_height)
                    txt = self.extract_text_from_backend(page0, bbox)
                    element.text = txt
                    element.orig = txt

        return conv_res.document

    def _turn_md_into_doc(self, conv_res):
        def _extract_markdown_code(text):
            """
            Extracts text from markdown code blocks (enclosed in triple backticks).
            If no code blocks are found, returns the original text.

            Args:
                text (str): Input text that may contain markdown code blocks

            Returns:
                str: Extracted code if code blocks exist, otherwise original text
            """
            # Regex pattern to match content between triple backticks
            # This handles multiline content and optional language specifier
            pattern = r"^```(?:\w*\n)?(.*?)```(\n)*$"

            # Search with DOTALL flag to match across multiple lines
            mtch = re.search(pattern, text, re.DOTALL)

            if mtch:
                # Return only the content of the first capturing group
                return mtch.group(1)
            else:
                # No code blocks found, return original text
                return text

        for pg_idx, page in enumerate(conv_res.pages):
            page_no = pg_idx + 1  # FIXME: might be incorrect

            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text + "\n\n"

            predicted_text = _extract_markdown_code(text=predicted_text)

            response_bytes = BytesIO(predicted_text.encode("utf8"))
            out_doc = InputDocument(
                path_or_stream=response_bytes,
                filename=conv_res.input.file.name,
                format=InputFormat.MD,
                backend=MarkdownDocumentBackend,
            )
            backend = MarkdownDocumentBackend(
                in_doc=out_doc,
                path_or_stream=response_bytes,
            )
            page_doc = backend.convert()

            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            conv_res.document.add_page(
                page_no=page_no,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

            for item, level in page_doc.iterate_items():
                item.prov = [
                    ProvenanceItem(
                        page_no=pg_idx + 1,
                        bbox=BoundingBox(
                            t=0.0, b=0.0, l=0.0, r=0.0
                        ),  # FIXME: would be nice not to have to "fake" it
                        charspan=[0, 0],
                    )
                ]
                conv_res.document.append_child_item(child=item)

        return conv_res.document

    def _turn_html_into_doc(self, conv_res):
        def _extract_html_code(text):
            """
            Extracts text from markdown code blocks (enclosed in triple backticks).
            If no code blocks are found, returns the original text.

            Args:
                text (str): Input text that may contain markdown code blocks

            Returns:
                str: Extracted code if code blocks exist, otherwise original text
            """
            # Regex pattern to match content between triple backticks
            # This handles multiline content and optional language specifier
            pattern = r"^```(?:\w*\n)?(.*?)```(\n)*$"

            # Search with DOTALL flag to match across multiple lines
            mtch = re.search(pattern, text, re.DOTALL)

            if mtch:
                # Return only the content of the first capturing group
                return mtch.group(1)
            else:
                # No code blocks found, return original text
                return text

        for pg_idx, page in enumerate(conv_res.pages):
            page_no = pg_idx + 1  # FIXME: might be incorrect

            predicted_text = ""
            if page.predictions.vlm_response:
                predicted_text = page.predictions.vlm_response.text + "\n\n"

            predicted_text = _extract_html_code(text=predicted_text)

            response_bytes = BytesIO(predicted_text.encode("utf8"))
            out_doc = InputDocument(
                path_or_stream=response_bytes,
                filename=conv_res.input.file.name,
                format=InputFormat.MD,
                backend=HTMLDocumentBackend,
            )
            backend = HTMLDocumentBackend(
                in_doc=out_doc,
                path_or_stream=response_bytes,
            )
            page_doc = backend.convert()

            if page.image is not None:
                pg_width = page.image.width
                pg_height = page.image.height
            else:
                pg_width = 1
                pg_height = 1

            conv_res.document.add_page(
                page_no=page_no,
                size=Size(width=pg_width, height=pg_height),
                image=ImageRef.from_pil(image=page.image, dpi=72)
                if page.image
                else None,
            )

            for item, level in page_doc.iterate_items():
                item.prov = [
                    ProvenanceItem(
                        page_no=pg_idx + 1,
                        bbox=BoundingBox(
                            t=0.0, b=0.0, l=0.0, r=0.0
                        ),  # FIXME: would be nice not to have to "fake" it
                        charspan=[0, 0],
                    )
                ]
                conv_res.document.append_child_item(child=item)

        return conv_res.document

    @classmethod
    def get_default_options(cls) -> VlmPipelineOptions:
        return VlmPipelineOptions()

    @classmethod
    def is_backend_supported(cls, backend: AbstractDocumentBackend):
        return isinstance(backend, PdfDocumentBackend)
