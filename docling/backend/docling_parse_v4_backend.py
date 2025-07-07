import logging
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pypdfium2 as pdfium
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import SegmentedPdfPage, TextCell
from docling_parse.pdf_parser import DoclingPdfParser, PdfDocument
from PIL import Image
from pypdfium2 import PdfPage

from docling.backend.pdf_backend import PdfDocumentBackend, PdfPageBackend
from docling.datamodel.base_models import Size
from docling.utils.locks import pypdfium2_lock

if TYPE_CHECKING:
    from docling.datamodel.document import InputDocument

_log = logging.getLogger(__name__)


class DoclingParseV4PageBackend(PdfPageBackend):
    def __init__(self, parsed_page: SegmentedPdfPage, page_obj: PdfPage):
        self._ppage = page_obj
        self._dpage = parsed_page
        self.valid = parsed_page is not None
        # Cache page size
        self._cached_size = None

    def is_valid(self) -> bool:
        return self.valid

    def get_text_in_rect(self, bbox: BoundingBox) -> str:
        """Returns concatenated text of cells where bbox overlaps more than 0.5"""
        # Compute page size and scale once
        page_size = self.get_size()
        page_height = page_size.height
        scale = (
            1  # FIX - replace with param in get_text_in_rect across backends (optional)
        )
        # Precompute results
        results = []
        # Only call bbox.to_top_left_origin/scale ONCE per cell
        bbox_cache = {}
        # Precompute bbox to use as target for intersection
        target_bbox = bbox
        textline_cells = self._dpage.textline_cells

        for cell in textline_cells:
            # Key by cell id to avoid repeat allocations if needed
            if id(cell) not in bbox_cache:
                cell_bbox = cell.rect.to_bounding_box()
                # Optimization - many bounding box transformations are chained, fuse them to a single function if possible
                cell_bbox = cell_bbox.to_top_left_origin(page_height=page_height)
                if scale != 1:
                    cell_bbox = cell_bbox.scaled(scale)
                bbox_cache[id(cell)] = cell_bbox
            else:
                cell_bbox = bbox_cache[id(cell)]
            overlap_frac = cell_bbox.intersection_over_self(target_bbox)
            if overlap_frac > 0.5:
                results.append(cell.text)
        # String concatenation was previously quadratic; now it's linear
        return " ".join(results)

    def get_segmented_page(self) -> Optional[SegmentedPdfPage]:
        return self._dpage

    def get_text_cells(self) -> Iterable[TextCell]:
        return self._dpage.textline_cells

    def get_bitmap_rects(self, scale: float = 1) -> Iterable[BoundingBox]:
        AREA_THRESHOLD = 0  # 32 * 32

        images = self._dpage.bitmap_resources

        for img in images:
            cropbox = img.rect.to_bounding_box().to_top_left_origin(
                self.get_size().height
            )

            if cropbox.area() > AREA_THRESHOLD:
                cropbox = cropbox.scaled(scale=scale)

                yield cropbox

    def get_page_image(
        self, scale: float = 1, cropbox: Optional[BoundingBox] = None
    ) -> Image.Image:
        page_size = self.get_size()

        if not cropbox:
            cropbox = BoundingBox(
                l=0,
                r=page_size.width,
                t=0,
                b=page_size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
            padbox = BoundingBox(
                l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT
            )
        else:
            padbox = cropbox.to_bottom_left_origin(page_size.height).model_copy()
            padbox.r = page_size.width - padbox.r
            padbox.t = page_size.height - padbox.t

        with pypdfium2_lock:
            image = (
                self._ppage.render(
                    scale=scale * 1.5,
                    rotation=0,  # no additional rotation
                    crop=padbox.as_tuple(),
                )
                .to_pil()
                .resize(
                    size=(round(cropbox.width * scale), round(cropbox.height * scale))
                )
            )  # We resize the image from 1.5x the given scale to make it sharper.

        return image

    def get_size(self) -> Size:
        # Cache the page size to avoid repeated expensive calls/locking
        if self._cached_size is None:
            with pypdfium2_lock:
                self._cached_size = Size(
                    width=self._ppage.get_width(), height=self._ppage.get_height()
                )
        return self._cached_size
        # TODO: Take width and height from docling-parse.
        # return Size(
        #    width=self._dpage.dimension.width,
        #    height=self._dpage.dimension.height,
        # )

    def unload(self):
        self._ppage = None
        self._dpage = None


class DoclingParseV4DocumentBackend(PdfDocumentBackend):
    def __init__(self, in_doc: "InputDocument", path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        with pypdfium2_lock:
            self._pdoc = pdfium.PdfDocument(self.path_or_stream)
        self.parser = DoclingPdfParser(loglevel="fatal")
        self.dp_doc: PdfDocument = self.parser.load(path_or_stream=self.path_or_stream)
        success = self.dp_doc is not None

        if not success:
            raise RuntimeError(
                f"docling-parse v4 could not load document {self.document_hash}."
            )

    def page_count(self) -> int:
        # return len(self._pdoc)  # To be replaced with docling-parse API

        len_1 = len(self._pdoc)
        len_2 = self.dp_doc.number_of_pages()

        if len_1 != len_2:
            _log.error(f"Inconsistent number of pages: {len_1}!={len_2}")

        return len_2

    def load_page(
        self, page_no: int, create_words: bool = True, create_textlines: bool = True
    ) -> DoclingParseV4PageBackend:
        with pypdfium2_lock:
            seg_page = self.dp_doc.get_page(
                page_no + 1,
                create_words=create_words,
                create_textlines=create_textlines,
            )

            # In Docling, all TextCell instances are expected with top-left origin.
            [
                tc.to_top_left_origin(seg_page.dimension.height)
                for tc in seg_page.textline_cells
            ]
            [
                tc.to_top_left_origin(seg_page.dimension.height)
                for tc in seg_page.char_cells
            ]
            [
                tc.to_top_left_origin(seg_page.dimension.height)
                for tc in seg_page.word_cells
            ]

            return DoclingParseV4PageBackend(
                seg_page,
                self._pdoc[page_no],
            )

    def is_valid(self) -> bool:
        return self.page_count() > 0

    def unload(self):
        super().unload()
        # Unload docling-parse document first
        if self.dp_doc is not None:
            self.dp_doc.unload()
            self.dp_doc = None

        # Then close pypdfium2 document with proper locking
        if self._pdoc is not None:
            with pypdfium2_lock:
                try:
                    self._pdoc.close()
                except Exception:
                    # Ignore cleanup errors
                    pass
            self._pdoc = None
