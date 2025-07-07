import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Final, Set, Union

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    GroupItem,
    GroupLabel,
    ImageRef,
    Size,
    TableCell,
    TableData,
)

from docling.backend.abstract_backend import DeclarativeDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument

DEFAULT_IMAGE_WIDTH: Final = 128
DEFAULT_IMAGE_HEIGHT: Final = 128

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_WIDTH: Final = 128
DEFAULT_IMAGE_HEIGHT: Final = 128


class AsciiDocBackend(DeclarativeDocumentBackend):
    def __init__(self, in_doc: InputDocument, path_or_stream: Union[BytesIO, Path]):
        super().__init__(in_doc, path_or_stream)

        self.path_or_stream = path_or_stream

        try:
            if isinstance(self.path_or_stream, BytesIO):
                text_stream = self.path_or_stream.getvalue().decode("utf-8")
                self.lines = text_stream.split("\n")
            if isinstance(self.path_or_stream, Path):
                with open(self.path_or_stream, encoding="utf-8") as f:
                    self.lines = f.readlines()
            self.valid = True

        except Exception as e:
            raise RuntimeError(
                f"Could not initialize AsciiDoc backend for file with hash {self.document_hash}."
            ) from e
        return

    def is_valid(self) -> bool:
        return self.valid

    @classmethod
    def supports_pagination(cls) -> bool:
        return False

    def unload(self):
        return

    @classmethod
    def supported_formats(cls) -> Set[InputFormat]:
        return {InputFormat.ASCIIDOC}

    def convert(self) -> DoclingDocument:
        """
        Parses the ASCII into a structured document model.
        """

        origin = DocumentOrigin(
            filename=self.file.name or "file",
            mimetype="text/asciidoc",
            binary_hash=self.document_hash,
        )

        doc = DoclingDocument(name=self.file.stem or "file", origin=origin)

        doc = self._parse(doc)

        return doc

    def _parse(self, doc: DoclingDocument):
        """
        Main function that orchestrates the parsing by yielding components:
        title, section headers, text, lists, and tables.
        """
        in_list = False
        in_table = False

        text_data = []
        table_data = []
        caption_data = []

        parents: dict[int, Union[GroupItem, None]] = {i: None for i in range(10)}
        indents: dict[int, Union[GroupItem, None]] = {i: None for i in range(10)}

        append_text = text_data.append
        append_caption = caption_data.append
        get_current_level = self._get_current_level
        get_current_parent = self._get_current_parent

        strip = str.strip

        iter_lines = self.lines
        for line in iter_lines:
            # --------------- Title
            if _RE_TITLE.match(line):
                item = self._parse_title(line)
                level = item["level"]
                parents[level] = doc.add_text(
                    text=item["text"], label=DocItemLabel.TITLE
                )
                continue

            # --------------- Section headers
            if _RE_SECTION_HEADER.match(line):
                item = self._parse_section_header(line)
                level = item["level"]
                parents[level] = doc.add_heading(
                    text=item["text"], level=level, parent=parents[level - 1]
                )
                for k in range(level + 1, 10):
                    parents[k] = None
                continue

            # --------------- Lists
            if _RE_LIST_ITEM.match(line):
                item = self._parse_list_item(line)
                level = get_current_level(parents)

                if not in_list:
                    in_list = True
                    parents[level + 1] = doc.add_group(
                        parent=parents[level], name="list", label=GroupLabel.LIST
                    )
                    indents[level + 1] = item["indent"]

                elif in_list and item["indent"] > indents[level]:
                    parents[level + 1] = doc.add_group(
                        parent=parents[level], name="list", label=GroupLabel.LIST
                    )
                    indents[level + 1] = item["indent"]

                elif in_list and item["indent"] < indents[level]:
                    while item["indent"] < indents[level]:
                        parents[level] = None
                        indents[level] = None
                        level -= 1

                doc.add_list_item(item["text"], parent=get_current_parent(parents))
                continue

            elif in_list and not _RE_LIST_ITEM.match(line):
                in_list = False
                level = get_current_level(parents)
                parents[level] = None

            # --------------- Tables
            ls = strip(line)
            if ls == "|===" and not in_table:
                in_table = True
                continue

            if in_table and (not _RE_TABLE_LINE.match(line) or ls == "|==="):
                caption = None
                if caption_data:
                    caption = doc.add_text(
                        text=" ".join(caption_data), label=DocItemLabel.CAPTION
                    )
                caption_data.clear()
                if table_data:
                    data = self._populate_table_as_grid(table_data)
                    doc.add_table(
                        data=data, parent=get_current_parent(parents), caption=caption
                    )
                    table_data.clear()
                in_table = False
                continue

            if _RE_TABLE_LINE.match(line):
                in_table = True
                table_data.append(self._parse_table_line(line))
                continue

            # --------------- Picture
            if _RE_PICTURE.match(line):
                caption = None
                if caption_data:
                    caption = doc.add_text(
                        text=" ".join(caption_data), label=DocItemLabel.CAPTION
                    )
                caption_data.clear()
                item = self._parse_picture(line)

                width = int(item.get("width", DEFAULT_IMAGE_WIDTH))
                height = int(item.get("height", DEFAULT_IMAGE_HEIGHT))
                size = Size(width=width, height=height)

                uri = None
                if "uri" in item and not item["uri"].startswith("http"):
                    if item["uri"].startswith("//"):
                        uri = "file:" + item["uri"]
                    elif item["uri"].startswith("/"):
                        uri = "file:/" + item["uri"]
                    else:
                        uri = "file://" + item["uri"]

                image = ImageRef(mimetype="image/png", size=size, dpi=70, uri=uri)
                doc.add_picture(image=image, caption=caption)
                continue

            # --------------- Caption
            if not caption_data and _RE_CAPTION.match(line):
                item = self._parse_caption(line)
                append_caption(item["text"])
                continue

            elif ls and caption_data:  # multiline caption
                item = self._parse_text(line)
                append_caption(item["text"])
                continue

            # --------------- Plain text
            if not ls and text_data:
                doc.add_text(
                    text=" ".join(text_data),
                    label=DocItemLabel.PARAGRAPH,
                    parent=get_current_parent(parents),
                )
                text_data.clear()
                continue
            elif ls:  # allow multiline texts
                item = self._parse_text(line)
                append_text(item["text"])

        if text_data:
            doc.add_text(
                text=" ".join(text_data),
                label=DocItemLabel.PARAGRAPH,
                parent=get_current_parent(parents),
            )
            text_data.clear()

        if in_table and table_data:
            data = self._populate_table_as_grid(table_data)
            doc.add_table(data=data, parent=get_current_parent(parents))
            table_data.clear()

        return doc

    @staticmethod
    def _get_current_level(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return k - 1

        return 0

    @staticmethod
    def _get_current_parent(parents):
        for k, v in parents.items():
            if v is None and k > 0:
                return parents[k - 1]

        return None

    #   =========   Title
    @staticmethod
    def _is_title(line):
        return re.match(r"^= ", line)

    @staticmethod
    def _parse_title(line):
        return {"type": "title", "text": line[2:].strip(), "level": 0}

    #   =========   Section headers
    @staticmethod
    def _is_section_header(line):
        return re.match(r"^==+\s+", line)

    @staticmethod
    def _parse_section_header(line):
        match = re.match(r"^(=+)\s+(.*)", line)
        marker = match.group(1)
        text = match.group(2)
        header_level = marker.count("=")
        return {
            "type": "header",
            "level": header_level - 1,
            "text": text.strip(),
        }

    #   =========   Lists
    @staticmethod
    def _is_list_item(line):
        return re.match(r"^(\s)*(\*|-|\d+\.|\w+\.) ", line)

    @staticmethod
    def _parse_list_item(line):
        """Extract the item marker (number or bullet symbol) and the text of the item."""
        match = _RE_LIST_ITEM_PARSE.match(line)
        if match:
            indent = match.group(1)
            marker = match.group(2)
            text = match.group(3)
            numbered = marker not in ("*", "-")
            return {
                "type": "list_item",
                "marker": marker,
                "text": text.strip(),
                "numbered": numbered,
                "indent": 0 if not indent else len(indent),
            }
        else:
            # Fallback if no match
            return {
                "type": "list_item",
                "marker": "-",
                "text": line,
                "numbered": False,
                "indent": 0,
            }

    #   =========   Tables
    @staticmethod
    def _is_table_line(line):
        return re.match(r"^\|.*\|", line)

    @staticmethod
    def _parse_table_line(line):
        # Split table cells and trim extra spaces
        return [cell.strip() for cell in line.split("|") if cell.strip()]

    @staticmethod
    def _populate_table_as_grid(table_data):
        num_rows = len(table_data)
        num_cols = max((len(row) for row in table_data), default=0)
        data = TableData(num_rows=num_rows, num_cols=num_cols, table_cells=[])
        for row_idx, row in enumerate(table_data):
            for col_idx, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    row_span=1,
                    col_span=1,
                    start_row_offset_idx=row_idx,
                    end_row_offset_idx=row_idx + 1,
                    start_col_offset_idx=col_idx,
                    end_col_offset_idx=col_idx + 1,
                    column_header=(row_idx == 0),
                    row_header=False,
                )
                data.table_cells.append(cell)
        return data

    #   =========   Pictures
    @staticmethod
    def _is_picture(line):
        return re.match(r"^image::", line)

    @staticmethod
    def _parse_picture(line):
        """
        Parse an image macro, extracting its path and attributes.
        Syntax: image::path/to/image.png[Alt Text, width=200, height=150, align=center]
        """
        mtch = _RE_PICTURE_PARSE.match(line)
        if mtch:
            picture_path = mtch.group(1).strip()
            attributes = mtch.group(2).split(",")
            picture_info = {"type": "picture", "uri": picture_path}
            if attributes:
                picture_info["alt"] = attributes[0].strip() if attributes[0] else ""
                for attr in attributes[1:]:
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        picture_info[key.strip()] = value.strip()
            return picture_info
        return {"type": "picture", "uri": line}

    #   =========   Captions
    @staticmethod
    def _is_caption(line):
        return re.match(r"^\.(.+)", line)

    @staticmethod
    def _parse_caption(line):
        mtch = _RE_CAPTION.match(line)
        if mtch:
            text = mtch.group(1)
            return {"type": "caption", "text": text}
        return {"type": "caption", "text": ""}

    #   =========   Plain text
    @staticmethod
    def _parse_text(line):
        return {"type": "text", "text": line.strip()}


_RE_TITLE = re.compile(r"^= ")

_RE_SECTION_HEADER = re.compile(r"^==+\s+")

_RE_LIST_ITEM = re.compile(r"^(\s)*(\*|-|\d+\.|\w+\.) ")

_RE_LIST_ITEM_PARSE = re.compile(r"^(\s*)(\*|-|\d+\.)\s+(.*)")

_RE_TABLE_LINE = re.compile(r"^\|.*\|")

_RE_PICTURE = re.compile(r"^image::")

_RE_PICTURE_PARSE = re.compile(r"^image::(.+)\[(.*)\]$")

_RE_CAPTION = re.compile(r"^\.(.+)")
