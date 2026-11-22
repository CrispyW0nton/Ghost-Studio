"""Qt-free KOTOR narrative data authoring contracts.

This module is a clean-room implementation of the data workflows exposed by
Ghost Scripter.  It deliberately depends on PyKotor's public format APIs where
they preserve the required data, while retaining TLK metadata that PyKotor's
high-level model does not currently expose.

The services prove structural read/write compatibility.  They do not claim a
retail KOTOR runtime proof; built resources still require the normal module or
Override install-and-test workflow.
"""

from __future__ import annotations

import configparser
import io
import os
import re
import struct
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4


_TLK_HEADER = struct.Struct("<4s4sIII")
_TLK_ENTRY = struct.Struct("<I16sIIIIf")
_TLK_MAGIC = b"TLK "
_TLK_VERSION = b"V3.0"
_TLK_HAS_TEXT = 0x01
_TLK_HAS_SOUND = 0x02
_TLK_HAS_SOUND_LENGTH = 0x04
_UNSET = object()


def _read_source(source: bytes | bytearray | memoryview | str | Path) -> tuple[bytes, Path | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), None
    path = Path(source)
    return path.read_bytes(), path


def _atomic_write(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(bytes(data))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class DataDiagnostic:
    """A format-agnostic validation result for a narrative data document."""

    severity: str
    code: str
    message: str
    item: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity.casefold() in {"blocking", "error"}


# ---------------------------------------------------------------------------
# 2DA tables and TSLPatcher diffs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoDASnapshot:
    headers: tuple[str, ...]
    labels: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def copy_binary_2da_cells(
    source: bytes, edits: Iterable[tuple[int, str, int, str]]
) -> bytes:
    """Copy existing V2.b cell values without rebuilding the string pool.

    Each edit is (destination row, destination column, source row, source
    column). All donors refer to the original table. This supports targeted
    edits when a complete re-encode would overflow the 16-bit cell offsets.
    """
    signature = b"2DA V2.b\n"
    if not source.startswith(signature):
        raise ValueError("Cell copying requires a binary V2.b 2DA.")
    before = TwoDADocument.load(source).snapshot()
    table = bytearray(source)
    start = source.index(b"\0", len(signature)) + 1
    row_count, = struct.unpack_from("<I", source, start)
    if row_count != len(before.rows):
        raise ValueError("Binary 2DA row count differs from the decoded table.")
    start += 4
    for _ in range(row_count):
        start = source.index(b"\t", start) + 1
    width = len(before.headers)
    columns = {name: i for i, name in enumerate(before.headers)}
    if start + 2 * (row_count * width + 1) > len(source):
        raise ValueError("Truncated binary 2DA cell index.")
    applied = set()
    for dst_row, dst_column, src_row, src_column in edits:
        if not (0 <= dst_row < row_count and 0 <= src_row < row_count):
            raise IndexError("Binary 2DA row outside the table.")
        dst = start + 2 * (dst_row * width + columns[dst_column])
        src = start + 2 * (src_row * width + columns[src_column])
        if dst in applied:
            raise ValueError("Duplicate binary 2DA destination cell.")
        table[dst:dst+2] = source[src:src+2]
        applied.add(dst)
    return bytes(table)


class TwoDADocument:
    """Editable 2DA table with immutable, undo-ready snapshots."""

    def __init__(
        self,
        headers: Sequence[str] = (),
        labels: Sequence[str] = (),
        rows: Sequence[Sequence[object]] = (),
        *,
        source_path: str | Path | None = None,
    ) -> None:
        self._headers = [str(value) for value in headers]
        self._labels = [str(value) for value in labels]
        self._rows = [[str(value) for value in row] for row in rows]
        self.source_path = Path(source_path) if source_path is not None else None
        self._assert_shape()

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "TwoDADocument":
        from pykotor.resource.formats.twoda import read_2da

        data, source_path = _read_source(source)
        if data.startswith((b"2DA V2.0\n", b"2DA V2.0\r\n", b"2DA V2.0\r")):
            # PyKotor's reader is binary-only; loose Override tables are often ASCII V2.0.
            from src.core.templates.twoda import TwoDA

            text_table = TwoDA.from_bytes(data)
            # "****" is ASCII's blank marker; binary V2.b stores the same cell as "".
            rows = [
                ["" if cell == "****" else cell for cell in row]
                for row in text_table._rows
            ]
            return cls(text_table.columns, list(text_table._labels), rows, source_path=source_path)
        table = read_2da(data)
        recovered_legacy_binary_cells = False
        # Some shipped V2.b tables declare a zero-length string pool yet keep
        # offset-addressed strings; the Kaitai path then returns only empty cells.
        # Use PyKotor's offset-following legacy decoder only when it recovers data
        # with the identical table shape.
        if data.startswith(b"2DA V2.b") and table.get_height() and not any(
            table.get_cell(i, header)
            for i in range(table.get_height())
            for header in table.get_headers()
        ):
            from pykotor.common.stream import BinaryReader
            from pykotor.resource.formats.twoda.io_twoda import _load_twoda_legacy

            legacy = _load_twoda_legacy(BinaryReader.from_bytes(data))
            if legacy.get_headers() != table.get_headers() or legacy.get_labels() != table.get_labels():
                raise ValueError("Binary 2DA readers disagree on table shape.")
            if any(legacy.get_cell(i, header) for i in range(legacy.get_height()) for header in legacy.get_headers()):
                table = legacy
                recovered_legacy_binary_cells = True
        headers = table.get_headers()
        labels = table.get_labels()
        rows = [
            [table.get_cell(row_index, header) for header in headers]
            for row_index in range(table.get_height())
        ]
        document = cls(headers, labels, rows, source_path=source_path)
        document.recovered_legacy_binary_cells = recovered_legacy_binary_cells
        return document

    @classmethod
    def from_snapshot(cls, snapshot: TwoDASnapshot) -> "TwoDADocument":
        return cls(snapshot.headers, snapshot.labels, snapshot.rows)

    @property
    def headers(self) -> tuple[str, ...]:
        return tuple(self._headers)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._labels)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def column_count(self) -> int:
        return len(self._headers)

    def _assert_shape(self) -> None:
        if len(self._labels) != len(self._rows):
            raise ValueError("2DA row labels and rows must have the same length.")
        for row in self._rows:
            if len(row) != len(self._headers):
                raise ValueError("Every 2DA row must contain one value per column.")

    def snapshot(self) -> TwoDASnapshot:
        return TwoDASnapshot(self.headers, self.labels, tuple(tuple(row) for row in self._rows))

    def restore(self, snapshot: TwoDASnapshot) -> None:
        restored = TwoDADocument.from_snapshot(snapshot)
        self._headers = list(restored._headers)
        self._labels = list(restored._labels)
        self._rows = [list(row) for row in restored._rows]

    def row(self, row_index: int) -> dict[str, str]:
        values = self._rows[row_index]
        return dict(zip(self._headers, values, strict=True))

    def cell(self, row_index: int, column: str) -> str:
        try:
            column_index = self._headers.index(str(column))
        except ValueError as exc:
            raise KeyError(f"Unknown 2DA column: {column}") from exc
        return self._rows[row_index][column_index]

    def set_cell(self, row_index: int, column: str, value: object) -> str:
        try:
            column_index = self._headers.index(str(column))
        except ValueError as exc:
            raise KeyError(f"Unknown 2DA column: {column}") from exc
        old_value = self._rows[row_index][column_index]
        self._rows[row_index][column_index] = "" if value is None else str(value)
        return old_value

    def set_row_label(self, row_index: int, label: object) -> str:
        new_label = str(label)
        if new_label in self._labels and self._labels[row_index] != new_label:
            raise ValueError(f"2DA row label already exists: {new_label}")
        old_label = self._labels[row_index]
        self._labels[row_index] = new_label
        return old_label

    def _next_label(self) -> str:
        numeric_labels = [int(label) for label in self._labels if label.isdecimal()]
        candidate = max(numeric_labels, default=-1) + 1
        while str(candidate) in self._labels:
            candidate += 1
        return str(candidate)

    def add_row(self, values: dict[str, object] | None = None, *, label: str | None = None) -> int:
        row_label = self._next_label() if label is None else str(label)
        if row_label in self._labels:
            raise ValueError(f"2DA row label already exists: {row_label}")
        supplied = values or {}
        unknown = set(supplied).difference(self._headers)
        if unknown:
            raise KeyError(f"Unknown 2DA columns: {', '.join(sorted(unknown))}")
        self._labels.append(row_label)
        self._rows.append(["" if supplied.get(header) is None else str(supplied.get(header, "")) for header in self._headers])
        return len(self._rows) - 1

    def copy_row(
        self,
        source_index: int,
        *,
        label: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> int:
        values = self.row(source_index)
        values.update(overrides or {})
        return self.add_row(values, label=label)

    def duplicate_rows(self, source_indices: Iterable[int]) -> tuple[int, ...]:
        """Append copies of the requested rows with fresh row labels.

        Source rows are captured before the table is changed, so duplicating a
        multi-row selection is deterministic even when the selection order is
        different from the table's storage order.
        """

        indices = tuple(dict.fromkeys(int(index) for index in source_indices))
        captured = [self.row(index) for index in indices]
        return tuple(self.add_row(values) for values in captured)

    def remove_row(self, row_index: int) -> tuple[str, dict[str, str]]:
        label = self._labels.pop(row_index)
        values = self._rows.pop(row_index)
        return label, dict(zip(self._headers, values, strict=True))

    def add_column(self, header: str, default: object = "") -> int:
        name = str(header).strip()
        if not name:
            raise ValueError("A 2DA column name cannot be empty.")
        if name in self._headers:
            raise ValueError(f"2DA column already exists: {name}")
        self._headers.append(name)
        value = "" if default is None else str(default)
        for row in self._rows:
            row.append(value)
        return len(self._headers) - 1

    def rename_column(self, header: str, new_header: str) -> int:
        """Rename a column without changing its position or cell values."""

        try:
            column_index = self._headers.index(str(header))
        except ValueError as exc:
            raise KeyError(f"Unknown 2DA column: {header}") from exc
        name = str(new_header).strip()
        if not name:
            raise ValueError("A 2DA column name cannot be empty.")
        if name in self._headers and self._headers[column_index] != name:
            raise ValueError(f"2DA column already exists: {name}")
        self._headers[column_index] = name
        return column_index

    def remove_column(self, header: str) -> tuple[int, tuple[str, ...]]:
        try:
            column_index = self._headers.index(str(header))
        except ValueError as exc:
            raise KeyError(f"Unknown 2DA column: {header}") from exc
        self._headers.pop(column_index)
        values = []
        for row in self._rows:
            values.append(row.pop(column_index))
        return column_index, tuple(values)

    def apply_cell_edits(
        self,
        edits: Iterable[tuple[int, str | None, object]],
    ) -> TwoDASnapshot:
        """Apply a clipboard-like group of edits as one atomic table change.

        ``None`` identifies the row-label column.  Building a candidate table
        before committing also permits a valid multi-row label swap while
        preventing a partially applied paste when any target is invalid.
        """

        before = self.snapshot()
        labels = list(self._labels)
        rows = [list(row) for row in self._rows]
        for raw_row, raw_column, value in edits:
            row_index = int(raw_row)
            if row_index < 0 or row_index >= len(rows):
                raise IndexError(f"2DA row index out of range: {row_index}")
            text = "" if value is None else str(value)
            if raw_column is None:
                labels[row_index] = text
                continue
            column = str(raw_column)
            try:
                column_index = self._headers.index(column)
            except ValueError as exc:
                raise KeyError(f"Unknown 2DA column: {column}") from exc
            rows[row_index][column_index] = text

        candidate = TwoDADocument(self._headers, labels, rows)
        if len(set(candidate.labels)) != candidate.row_count:
            raise ValueError("2DA row labels must remain unique after paste.")
        self.restore(candidate.snapshot())
        return before

    def search(self, query: str, *, columns: Iterable[str] | None = None) -> tuple[int, ...]:
        needle = str(query).casefold()
        selected = self._headers if columns is None else [str(column) for column in columns]
        indices = []
        for column in selected:
            if column not in self._headers:
                raise KeyError(f"Unknown 2DA column: {column}")
            indices.append(self._headers.index(column))
        matches = []
        for row_index, row in enumerate(self._rows):
            values = [self._labels[row_index], *(row[index] for index in indices)]
            if any(needle in value.casefold() for value in values):
                matches.append(row_index)
        return tuple(matches)

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics: list[DataDiagnostic] = []
        if not self._headers:
            diagnostics.append(DataDiagnostic("blocking", "2da.no_columns", "The 2DA table has no columns."))
        if len(set(self._headers)) != len(self._headers):
            diagnostics.append(DataDiagnostic("blocking", "2da.duplicate_columns", "2DA column names must be unique."))
        if len(set(self._labels)) != len(self._labels):
            diagnostics.append(DataDiagnostic("blocking", "2da.duplicate_labels", "2DA row labels must be unique."))
        if any(not header.strip() for header in self._headers):
            diagnostics.append(DataDiagnostic("blocking", "2da.empty_column", "2DA column names cannot be empty."))
        return tuple(diagnostics)

    def to_pykotor(self):
        from pykotor.resource.formats.twoda import TwoDA

        table = TwoDA(list(self._headers))
        for label, values in zip(self._labels, self._rows, strict=True):
            table.add_row(label, dict(zip(self._headers, values, strict=True)))
        return table

    def to_bytes(self, *, append_only_base: TwoDASnapshot | None = None) -> bytes:
        """Serialize, optionally retaining duplicate row labels inherited from a base."""
        from pykotor.resource.formats.twoda import bytes_2da

        inherited_labels = False
        if append_only_base is not None:
            base_count = len(append_only_base.rows)
            base = self.snapshot()
            if (base.headers != append_only_base.headers
                    or base.labels[:base_count] != append_only_base.labels
                    or base.rows[:base_count] != append_only_base.rows):
                raise ValueError("Append-only serialization requires the exact original table prefix.")
            added = base.labels[base_count:]
            if len(set(added)) != len(added) or set(added).intersection(append_only_base.labels):
                raise ValueError("New 2DA row labels must be unique.")
            inherited_labels = True

        blocking = [
            diagnostic.message for diagnostic in self.validate()
            if diagnostic.blocking and not (inherited_labels and diagnostic.code == "2da.duplicate_labels")
        ]
        if blocking:
            raise ValueError("Cannot serialize invalid 2DA: " + "; ".join(blocking))
        unique = {cell for row in self._rows for cell in row}
        encoded = {cell: cell.encode("windows-1252") for cell in unique}
        if any(b"\0" in data for data in encoded.values()):
            raise ValueError("2DA cells cannot contain a NUL byte.")
        if sum(len(data) + 1 for data in encoded.values()) > 65535:
            # V2.b addresses cells with uint16 pool offsets; share suffixes before failing.
            return self._compact_binary(encoded)
        return bytes(bytes_2da(self.to_pykotor()))

    def _compact_binary(self, encoded: dict[str, bytes]) -> bytes:
        """Serialize a large V2.b table with shared string suffixes in the uint16 cell pool."""
        from array import array
        import sys

        # Sorted by reversed bytes (descending), each string comes right before the
        # strings that are its suffixes, so they can point into the same host string.
        hosts = sorted(encoded, key=lambda value: encoded[value][::-1], reverse=True)
        pool = bytearray()
        offsets: dict[str, int] = {}
        previous = b""
        terminator = 0
        for value in hosts:
            raw = encoded[value]
            if not pool or not previous.endswith(raw):
                pool.extend(raw + b"\0")
                terminator = len(pool) - 1
            offsets[value] = terminator - len(raw)
            previous = raw
        if len(pool) > 65535:
            raise ValueError(f"2DA cell strings need {len(pool)} bytes even with shared suffixes (limit 65535).")
        indices = array("H", (offsets[cell] for row in self._rows for cell in row))
        if sys.byteorder != "little":
            indices.byteswap()
        headers = "".join(column + "\t" for column in self._headers).encode("windows-1252")
        labels = "".join(label + "\t" for label in self._labels).encode("windows-1252")
        return (b"2DA V2.b\n" + headers + b"\0" + struct.pack("<I", self.row_count) + labels
                + indices.tobytes() + struct.pack("<H", len(pool)) + pool)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError("A target path is required for a new 2DA document.")
        result = _atomic_write(target, self.to_bytes())
        self.source_path = result
        return result

    def export_changes_ini(self, original: "TwoDADocument", table_name: str) -> str:
        """Return a conservative TSLPatcher 2DA diff.

        TSLPatcher cannot safely express arbitrary row/column deletion or row
        label replacement.  Those cases are refused instead of producing a
        patch that silently clobbers another mod's changes.
        """

        missing_columns = [header for header in original.headers if header not in self._headers]
        if missing_columns:
            raise ValueError("TSLPatcher changes.ini cannot express DeleteColumn safely.")
        missing_labels = [label for label in original.labels if label not in self._labels]
        if missing_labels:
            raise ValueError("TSLPatcher changes.ini cannot express DeleteRow safely.")

        filename = Path(str(table_name)).name
        if not filename.casefold().endswith(".2da"):
            filename += ".2da"
        section_stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(filename).stem).strip("_") or "table"

        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser["2DAList"] = {"Table0": filename}
        parser[filename] = {}

        new_columns = [header for header in self._headers if header not in original.headers]
        for index, header in enumerate(new_columns):
            section = f"{section_stem}_add_column_{index}"
            parser[filename][f"AddColumn{index}"] = section
            parser[section] = {"ColumnLabel": header, "DefaultValue": "****"}

        original_by_label = {label: index for index, label in enumerate(original.labels)}
        changed_index = 0
        for label in original.labels:
            old_index = original_by_label[label]
            new_index = self._labels.index(label)
            edits: dict[str, str] = {}
            for header in self._headers:
                old_value = original.cell(old_index, header) if header in original.headers else "****"
                new_value = self.cell(new_index, header)
                if new_value != old_value:
                    edits[header] = new_value
            if edits:
                section = f"{section_stem}_change_row_{changed_index}"
                parser[filename][f"ChangeRow{changed_index}"] = section
                parser[section] = {"RowIndex": str(old_index), **edits}
                changed_index += 1

        added_index = 0
        for row_index, label in enumerate(self._labels):
            if label in original_by_label:
                continue
            section = f"{section_stem}_add_row_{added_index}"
            parser[filename][f"AddRow{added_index}"] = section
            parser[section] = {"RowLabel": label, **self.row(row_index)}
            added_index += 1

        output = io.StringIO()
        parser.write(output, space_around_delimiters=False)
        return output.getvalue()


# ---------------------------------------------------------------------------
# Talk tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TalkTableEntry:
    strref: int
    text: str = ""
    voiceover: str = ""
    sound_length: float = 0.0
    flags: int = 0
    volume_variance: int = 0
    pitch_variance: int = 0
    raw_text: bytes = field(default=b"", repr=False, compare=True)
    raw_voiceover: bytes = field(default=b"", repr=False, compare=True)


@dataclass(frozen=True)
class TalkTableSnapshot:
    language_id: int
    entries: tuple[TalkTableEntry, ...]
    trailing_data: bytes = b""


def _language_encoding(language_id: int) -> str:
    from pykotor.common.language import Language

    try:
        return Language(int(language_id)).get_encoding()
    except (TypeError, ValueError):
        return "cp1252"


class TalkTableDocument:
    """Editable TLK V3.0 document that retains entry metadata and flag bits."""

    def __init__(
        self,
        language_id: int = 0,
        entries: Sequence[TalkTableEntry] = (),
        *,
        trailing_data: bytes = b"",
        source_path: str | Path | None = None,
    ) -> None:
        self.language_id = int(language_id)
        self._entries = list(entries)
        self._trailing_data = bytes(trailing_data)
        self.source_path = Path(source_path) if source_path is not None else None

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "TalkTableDocument":
        data, source_path = _read_source(source)
        if len(data) < _TLK_HEADER.size:
            raise ValueError("TLK data is shorter than the V3.0 header.")
        magic, version, language_id, count, string_offset = _TLK_HEADER.unpack_from(data, 0)
        if magic != _TLK_MAGIC or version != _TLK_VERSION:
            raise ValueError("Expected a TLK V3.0 resource.")
        table_end = _TLK_HEADER.size + count * _TLK_ENTRY.size
        if table_end > len(data) or string_offset < table_end or string_offset > len(data):
            raise ValueError("TLK entry table or string-data offset is invalid.")

        encoding = _language_encoding(language_id)
        entries: list[TalkTableEntry] = []
        max_used = 0
        for index in range(count):
            offset = _TLK_HEADER.size + index * _TLK_ENTRY.size
            flags, sound_raw, volume, pitch, text_offset, text_size, sound_length = _TLK_ENTRY.unpack_from(data, offset)
            text_start = string_offset + text_offset
            text_end = text_start + text_size
            if text_start < string_offset or text_end > len(data):
                raise ValueError(f"TLK StrRef {index} points outside the string-data block.")
            raw_text = data[text_start:text_end]
            text = raw_text.decode(encoding, errors="replace")
            raw_voiceover = bytes(sound_raw)
            voiceover = raw_voiceover.split(b"\0", 1)[0].decode("ascii", errors="ignore")
            entries.append(
                TalkTableEntry(
                    strref=index,
                    text=text,
                    voiceover=voiceover,
                    sound_length=float(sound_length),
                    flags=int(flags),
                    volume_variance=int(volume),
                    pitch_variance=int(pitch),
                    raw_text=raw_text,
                    raw_voiceover=raw_voiceover,
                )
            )
            max_used = max(max_used, text_offset + text_size)
        trailing = data[string_offset + max_used :]

        # PyKotor remains the independent public-API structural parser.  The
        # local metadata view exists because its TLKEntry omits raw flags and
        # variance fields needed for a lossless edit.
        from pykotor.resource.formats.tlk import read_tlk

        try:
            read_tlk(data)
        except ValueError:
            # Some legacy TLKs contain non-ASCII garbage in the fixed-width
            # ResRef field.  The strict public parser rejects those resources;
            # the bounded parser above retains their raw bytes so an unrelated
            # text edit does not destroy recoverable metadata.
            pass
        return cls(language_id, entries, trailing_data=trailing, source_path=source_path)

    @property
    def entries(self) -> tuple[TalkTableEntry, ...]:
        return tuple(self._entries)

    def snapshot(self) -> TalkTableSnapshot:
        return TalkTableSnapshot(self.language_id, self.entries, self._trailing_data)

    def restore(self, snapshot: TalkTableSnapshot) -> None:
        self.language_id = int(snapshot.language_id)
        self._entries = list(snapshot.entries)
        self._trailing_data = bytes(snapshot.trailing_data)

    def entry(self, strref: int) -> TalkTableEntry:
        if strref < 0 or strref >= len(self._entries):
            raise IndexError(f"TLK StrRef is out of range: {strref}")
        return self._entries[strref]

    def add_entry(
        self,
        text: str = "",
        *,
        voiceover: str = "",
        sound_length: float = 0.0,
    ) -> int:
        strref = len(self._entries)
        flags = 0
        if text:
            flags |= _TLK_HAS_TEXT
        if voiceover:
            flags |= _TLK_HAS_SOUND
        if sound_length:
            flags |= _TLK_HAS_SOUND_LENGTH
        self._entries.append(
            TalkTableEntry(strref, str(text), str(voiceover), float(sound_length), flags)
        )
        return strref

    def update_entry(
        self,
        strref: int,
        *,
        text: object = _UNSET,
        voiceover: object = _UNSET,
        sound_length: object = _UNSET,
    ) -> TalkTableEntry:
        current = self.entry(strref)
        flags = current.flags
        changes: dict[str, Any] = {}
        if text is not _UNSET:
            text_value = str(text)
            flags = flags | _TLK_HAS_TEXT if text_value else flags & ~_TLK_HAS_TEXT
            changes.update(text=text_value, raw_text=b"")
        if voiceover is not _UNSET:
            voice_value = str(voiceover)
            encoded = voice_value.encode("ascii", errors="strict")
            if len(encoded) > 16:
                raise ValueError("A TLK voiceover ResRef cannot exceed 16 ASCII bytes.")
            flags = flags | _TLK_HAS_SOUND if voice_value else flags & ~_TLK_HAS_SOUND
            changes.update(voiceover=voice_value, raw_voiceover=b"")
        if sound_length is not _UNSET:
            length = float(sound_length)
            if length < 0:
                raise ValueError("TLK sound length cannot be negative.")
            flags = flags | _TLK_HAS_SOUND_LENGTH if length else flags & ~_TLK_HAS_SOUND_LENGTH
            changes["sound_length"] = length
        updated = replace(current, flags=flags, **changes)
        self._entries[strref] = updated
        return updated

    def search(self, query: str) -> tuple[int, ...]:
        needle = str(query).casefold()
        if not needle:
            return tuple(range(len(self._entries)))
        return tuple(
            entry.strref
            for entry in self._entries
            if needle in entry.text.casefold()
            or needle in entry.voiceover.casefold()
            or needle in str(entry.strref)
        )

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics: list[DataDiagnostic] = []
        for index, entry in enumerate(self._entries):
            if entry.strref != index:
                diagnostics.append(
                    DataDiagnostic("blocking", "tlk.strref_order", "TLK StrRefs must match entry positions.", str(index))
                )
            try:
                encoded_voice = entry.voiceover.encode("ascii", errors="strict")
            except UnicodeEncodeError:
                encoded_voice = b""
                diagnostics.append(
                    DataDiagnostic("blocking", "tlk.voiceover_ascii", "TLK voiceover ResRefs must be ASCII.", str(index))
                )
            if len(encoded_voice) > 16:
                diagnostics.append(
                    DataDiagnostic("blocking", "tlk.voiceover_length", "TLK voiceover ResRefs cannot exceed 16 bytes.", str(index))
                )
            if entry.sound_length < 0:
                diagnostics.append(
                    DataDiagnostic("blocking", "tlk.negative_length", "TLK sound length cannot be negative.", str(index))
                )
        return tuple(diagnostics)

    def _text_bytes(self, entry: TalkTableEntry, encoding: str) -> bytes:
        if entry.raw_text:
            try:
                if entry.raw_text.decode(encoding, errors="replace") == entry.text:
                    return entry.raw_text
            except (LookupError, UnicodeError):
                pass
        return entry.text.encode(encoding, errors="strict")

    @staticmethod
    def _voiceover_bytes(entry: TalkTableEntry) -> bytes:
        if entry.raw_voiceover:
            decoded = entry.raw_voiceover.split(b"\0", 1)[0].decode("ascii", errors="ignore")
            if decoded == entry.voiceover:
                return entry.raw_voiceover[:16].ljust(16, b"\0")
        return entry.voiceover.encode("ascii", errors="strict")[:16].ljust(16, b"\0")

    def to_bytes(self) -> bytes:
        blocking = [diagnostic.message for diagnostic in self.validate() if diagnostic.blocking]
        if blocking:
            raise ValueError("Cannot serialize invalid TLK: " + "; ".join(blocking))
        encoding = _language_encoding(self.language_id)
        encoded_text = [self._text_bytes(entry, encoding) for entry in self._entries]
        string_offset = _TLK_HEADER.size + len(self._entries) * _TLK_ENTRY.size
        string_data = bytearray()
        entry_table = bytearray()
        for entry, text_data in zip(self._entries, encoded_text, strict=True):
            text_offset = len(string_data)
            string_data.extend(text_data)
            entry_table.extend(
                _TLK_ENTRY.pack(
                    entry.flags & 0xFFFFFFFF,
                    self._voiceover_bytes(entry),
                    entry.volume_variance & 0xFFFFFFFF,
                    entry.pitch_variance & 0xFFFFFFFF,
                    text_offset,
                    len(text_data),
                    float(entry.sound_length),
                )
            )
        output = bytes(
            _TLK_HEADER.pack(_TLK_MAGIC, _TLK_VERSION, self.language_id, len(self._entries), string_offset)
            + entry_table
            + string_data
            + self._trailing_data
        )

        from pykotor.resource.formats.tlk import read_tlk

        try:
            read_tlk(output)
        except ValueError:
            # See load(): raw legacy ResRef bytes may intentionally remain
            # outside PyKotor's strict ASCII model.
            pass
        return output

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError("A target path is required for a new TLK document.")
        result = _atomic_write(target, self.to_bytes())
        self.source_path = result
        return result


# ---------------------------------------------------------------------------
# Journal / quest data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalizedText:
    stringref: int = -1
    substrings: tuple[tuple[int, str], ...] = ()

    @classmethod
    def from_english(cls, text: str, *, stringref: int = -1) -> "LocalizedText":
        return cls(int(stringref), ((0, str(text)),) if text else ())

    @classmethod
    def from_pykotor(cls, value: object) -> "LocalizedText":
        data = value.to_dict()  # type: ignore[attr-defined]
        substrings = tuple(sorted((int(key), str(text)) for key, text in data.get("substrings", {}).items()))
        return cls(int(data.get("stringref", -1)), substrings)

    def to_pykotor(self):
        from pykotor.common.language import LocalizedString

        return LocalizedString.from_dict(
            {"stringref": int(self.stringref), "substrings": {key: text for key, text in self.substrings}}
        )

    @property
    def english(self) -> str:
        values = dict(self.substrings)
        return values.get(0, next(iter(values.values()), ""))

    def with_english(self, text: str) -> "LocalizedText":
        values = dict(self.substrings)
        if text:
            values[0] = str(text)
        else:
            values.pop(0, None)
        return replace(self, substrings=tuple(sorted(values.items())))


@dataclass(frozen=True)
class JournalEntryRecord:
    entry_id: int
    text: LocalizedText = field(default_factory=LocalizedText)
    end: bool = False
    xp_percentage: float = 0.0
    source_index: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class JournalQuestRecord:
    tag: str
    name: LocalizedText = field(default_factory=LocalizedText)
    comment: str = ""
    planet_id: int = 0
    plot_index: int = 0
    priority: int = 4
    entries: tuple[JournalEntryRecord, ...] = ()
    source_index: int | None = field(default=None, repr=False)


@dataclass(frozen=True)
class JournalSnapshot:
    quests: tuple[JournalQuestRecord, ...]


class JournalDocument:
    """Editable JRL journal that preserves unknown retained GFF fields."""

    def __init__(
        self,
        quests: Sequence[JournalQuestRecord] = (),
        *,
        source_gff: object | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        self._quests = list(quests)
        self._source_gff = deepcopy(source_gff)
        self.source_path = Path(source_path) if source_path is not None else None

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "JournalDocument":
        from pykotor.resource.formats.gff import read_gff
        from pykotor.resource.generics.jrl import construct_jrl

        data, source_path = _read_source(source)
        gff = read_gff(data)
        with redirect_stdout(io.StringIO()):
            journal = construct_jrl(gff)
        quests: list[JournalQuestRecord] = []
        for quest_index, quest in enumerate(journal.quests):
            entries = tuple(
                JournalEntryRecord(
                    int(entry.entry_id),
                    LocalizedText.from_pykotor(entry.text),
                    bool(entry.end),
                    float(entry.xp_percentage),
                    entry_index,
                )
                for entry_index, entry in enumerate(quest.entries)
            )
            quests.append(
                JournalQuestRecord(
                    tag=str(quest.tag),
                    name=LocalizedText.from_pykotor(quest.name),
                    comment=str(quest.comment),
                    planet_id=int(quest.planet_id),
                    plot_index=int(quest.plot_index),
                    priority=int(quest.priority.value),
                    entries=entries,
                    source_index=quest_index,
                )
            )
        return cls(quests, source_gff=gff, source_path=source_path)

    @property
    def quests(self) -> tuple[JournalQuestRecord, ...]:
        return tuple(self._quests)

    def snapshot(self) -> JournalSnapshot:
        return JournalSnapshot(self.quests)

    def restore(self, snapshot: JournalSnapshot) -> None:
        self._quests = list(snapshot.quests)

    def add_quest(self, quest: JournalQuestRecord) -> int:
        self._quests.append(replace(quest, source_index=None))
        return len(self._quests) - 1

    def update_quest(self, quest_index: int, **changes: object) -> JournalQuestRecord:
        allowed = {"tag", "name", "comment", "planet_id", "plot_index", "priority"}
        unknown = set(changes).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported JRL quest fields: {', '.join(sorted(unknown))}")
        updated = replace(self._quests[quest_index], **changes)
        self._quests[quest_index] = updated
        return updated

    def remove_quest(self, quest_index: int) -> JournalQuestRecord:
        return self._quests.pop(quest_index)

    def add_entry(self, quest_index: int, entry: JournalEntryRecord) -> int:
        quest = self._quests[quest_index]
        entries = (*quest.entries, replace(entry, source_index=None))
        self._quests[quest_index] = replace(quest, entries=entries)
        return len(entries) - 1

    def update_entry(self, quest_index: int, entry_index: int, **changes: object) -> JournalEntryRecord:
        allowed = {"entry_id", "text", "end", "xp_percentage"}
        unknown = set(changes).difference(allowed)
        if unknown:
            raise KeyError(f"Unsupported JRL entry fields: {', '.join(sorted(unknown))}")
        quest = self._quests[quest_index]
        entries = list(quest.entries)
        entries[entry_index] = replace(entries[entry_index], **changes)
        self._quests[quest_index] = replace(quest, entries=tuple(entries))
        return entries[entry_index]

    def remove_entry(self, quest_index: int, entry_index: int) -> JournalEntryRecord:
        quest = self._quests[quest_index]
        entries = list(quest.entries)
        removed = entries.pop(entry_index)
        self._quests[quest_index] = replace(quest, entries=tuple(entries))
        return removed

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics: list[DataDiagnostic] = []
        seen_tags: set[str] = set()
        for quest_index, quest in enumerate(self._quests):
            item = quest.tag or f"quest {quest_index}"
            tag_key = quest.tag.casefold()
            if not quest.tag.strip():
                diagnostics.append(DataDiagnostic("blocking", "jrl.empty_tag", "Journal quest tags cannot be empty.", item))
            elif tag_key in seen_tags:
                diagnostics.append(DataDiagnostic("blocking", "jrl.duplicate_tag", "Journal quest tags must be unique.", item))
            seen_tags.add(tag_key)
            if quest.priority not in range(5):
                diagnostics.append(DataDiagnostic("blocking", "jrl.priority", "JRL priority must be between 0 and 4.", item))
            seen_ids: set[int] = set()
            for entry in quest.entries:
                if entry.entry_id < 0:
                    diagnostics.append(DataDiagnostic("blocking", "jrl.negative_entry", "JRL entry IDs cannot be negative.", item))
                if entry.entry_id in seen_ids:
                    diagnostics.append(DataDiagnostic("blocking", "jrl.duplicate_entry", "JRL entry IDs must be unique per quest.", item))
                seen_ids.add(entry.entry_id)
                if not 0.0 <= entry.xp_percentage <= 100.0:
                    diagnostics.append(DataDiagnostic("blocking", "jrl.xp_range", "JRL XP percentage must be from 0 to 100.", item))
        return tuple(diagnostics)

    def _source_categories(self):
        from pykotor.resource.formats.gff import GFFList

        if self._source_gff is None:
            return GFFList()
        return self._source_gff.root.acquire("Categories", GFFList())

    @staticmethod
    def _append_preserved_struct(target_list: object, source_struct: object) -> None:
        """Append a cloned GFF struct without PyKotor's per-field debug flood.

        PyKotor's public ``GFFStruct.merge`` currently emits one root DEBUG log
        for every copied field.  A vanilla global.jrl therefore produces
        thousands of console lines during one save.  ``GFFList.add`` is public
        and quiet; the field-map assignment is the smallest contained fallback
        until PyKotor exposes a quiet clone/append operation.
        """

        stored = target_list.add(source_struct.struct_id)  # type: ignore[attr-defined]
        stored._fields = deepcopy(source_struct._fields)  # noqa: SLF001  # type: ignore[attr-defined]

    def to_gff(self):
        from pykotor.resource.formats.gff import GFF, GFFContent, GFFList, GFFStruct

        blocking = [diagnostic.message for diagnostic in self.validate() if diagnostic.blocking]
        if blocking:
            raise ValueError("Cannot serialize invalid JRL: " + "; ".join(blocking))

        gff = deepcopy(self._source_gff) if self._source_gff is not None else GFF(GFFContent.JRL)
        source_categories = self._source_categories()
        categories = GFFList()
        for quest_index, quest in enumerate(self._quests):
            if quest.source_index is not None and 0 <= quest.source_index < len(source_categories):
                source_category = source_categories[quest.source_index]
                category = deepcopy(source_category)
                source_entries = source_category.acquire("EntryList", GFFList())
            else:
                category = GFFStruct(quest_index)
                source_entries = GFFList()
            category.set_string("Comment", quest.comment)
            category.set_locstring("Name", quest.name.to_pykotor())
            category.set_int32("PlanetID", int(quest.planet_id))
            category.set_int32("PlotIndex", int(quest.plot_index))
            category.set_uint32("Priority", int(quest.priority))
            category.set_string("Tag", quest.tag)

            entries = GFFList()
            for entry_index, entry in enumerate(quest.entries):
                if entry.source_index is not None and 0 <= entry.source_index < len(source_entries):
                    entry_struct = deepcopy(source_entries[entry.source_index])
                else:
                    entry_struct = GFFStruct(entry_index)
                entry_struct.set_uint16("End", int(bool(entry.end)))
                entry_struct.set_uint32("ID", int(entry.entry_id))
                entry_struct.set_locstring("Text", entry.text.to_pykotor())
                entry_struct.set_single("XP_Percentage", float(entry.xp_percentage))
                self._append_preserved_struct(entries, entry_struct)
            category.set_list("EntryList", entries)
            self._append_preserved_struct(categories, category)
        gff.root.set_list("Categories", categories)
        return gff

    def to_bytes(self) -> bytes:
        from pykotor.resource.formats.gff import bytes_gff, read_gff

        with redirect_stdout(io.StringIO()):
            output = bytes(bytes_gff(self.to_gff()))
            read_gff(output)
        return output

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError("A target path is required for a new JRL document.")
        result = _atomic_write(target, self.to_bytes())
        self.source_path = result
        return result


# ---------------------------------------------------------------------------
# globalcat.2da variable registry
# ---------------------------------------------------------------------------


GLOBAL_BOOLEAN = "Boolean"
GLOBAL_NUMBER = "Number"
GLOBAL_STRING = "String"
GLOBAL_LOCATION = "Location"
_GLOBAL_TYPES = {GLOBAL_BOOLEAN, GLOBAL_NUMBER, GLOBAL_STRING, GLOBAL_LOCATION}


@dataclass(frozen=True)
class GlobalVariableRecord:
    row_index: int
    row_label: str
    name: str
    value_type: str


class GlobalVariableTable:
    """Typed view over ``globalcat.2da``.

    The file registers variable names and types only.  Runtime Boolean, Number,
    String, and Location values live in game/save state and are not embedded in
    this table.
    """

    def __init__(self, table: TwoDADocument | None = None) -> None:
        self.table = table or TwoDADocument(("name", "type"))
        missing = {"name", "type"}.difference(self.table.headers)
        if missing:
            raise ValueError("globalcat.2da requires name and type columns.")

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "GlobalVariableTable":
        return cls(TwoDADocument.load(source))

    @property
    def variables(self) -> tuple[GlobalVariableRecord, ...]:
        return tuple(
            GlobalVariableRecord(index, self.table.labels[index], self.table.cell(index, "name"), self.table.cell(index, "type"))
            for index in range(self.table.row_count)
        )

    @staticmethod
    def _normalise_type(value_type: str) -> str:
        requested = str(value_type).casefold()
        for supported in _GLOBAL_TYPES:
            if supported.casefold() == requested:
                return supported
        raise ValueError("Global type must be Boolean, Number, String, or Location.")

    def add_variable(self, name: str, value_type: str, *, label: str | None = None) -> int:
        variable_name = str(name).strip()
        if not variable_name:
            raise ValueError("A KOTOR global variable name cannot be empty.")
        if any(variable.name.casefold() == variable_name.casefold() for variable in self.variables):
            raise ValueError(f"Global variable already exists: {variable_name}")
        return self.table.add_row(
            {"name": variable_name, "type": self._normalise_type(value_type)},
            label=label,
        )

    def update_variable(self, row_index: int, *, name: str | None = None, value_type: str | None = None) -> None:
        if name is not None:
            variable_name = str(name).strip()
            if not variable_name:
                raise ValueError("A KOTOR global variable name cannot be empty.")
            if any(
                variable.row_index != row_index and variable.name.casefold() == variable_name.casefold()
                for variable in self.variables
            ):
                raise ValueError(f"Global variable already exists: {variable_name}")
            self.table.set_cell(row_index, "name", variable_name)
        if value_type is not None:
            self.table.set_cell(row_index, "type", self._normalise_type(value_type))

    def remove_variable(self, row_index: int) -> GlobalVariableRecord:
        variable = self.variables[row_index]
        self.table.remove_row(row_index)
        return variable

    def snapshot(self) -> TwoDASnapshot:
        return self.table.snapshot()

    def restore(self, snapshot: TwoDASnapshot) -> None:
        self.table.restore(snapshot)

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics = list(self.table.validate())
        seen: set[str] = set()
        for variable in self.variables:
            key = variable.name.casefold()
            if not variable.name.strip():
                diagnostics.append(DataDiagnostic("blocking", "global.empty_name", "Global variable names cannot be empty.", variable.row_label))
            elif key in seen:
                diagnostics.append(DataDiagnostic("blocking", "global.duplicate_name", "Global variable names must be unique.", variable.name))
            seen.add(key)
            if variable.value_type not in _GLOBAL_TYPES:
                diagnostics.append(
                    DataDiagnostic(
                        "blocking",
                        "global.invalid_type",
                        "Global type must be Boolean, Number, String, or Location.",
                        variable.name,
                    )
                )
        return tuple(diagnostics)

    def to_bytes(self) -> bytes:
        blocking = [diagnostic.message for diagnostic in self.validate() if diagnostic.blocking]
        if blocking:
            raise ValueError("Cannot serialize invalid globalcat.2da: " + "; ".join(blocking))
        return self.table.to_bytes()

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.table.source_path
        if target is None:
            raise ValueError("A target path is required for a new globalcat.2da document.")
        result = _atomic_write(target, self.to_bytes())
        self.table.source_path = result
        return result

    def export_changes_ini(self, original: "GlobalVariableTable") -> str:
        return self.table.export_changes_ini(original.table, "globalcat.2da")


# ---------------------------------------------------------------------------
# SSF sound-set StrRef slots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoundSetSnapshot:
    stringrefs: tuple[int, ...]
    table_offset: int = 12
    header_padding: bytes = b""


class SoundSetDocument:
    """Editable KOTOR SSF V1.1 sound-set table.

    The first 28 entries have engine-defined names, but retail resources may
    contain additional meaningful entries.  Every stored entry is retained;
    new files use the common 40-entry table.  ``-1`` represents the on-disk
    UInt32 sentinel ``0xFFFFFFFF``.
    """

    def __init__(
        self,
        stringrefs: Sequence[int] | None = None,
        *,
        source_path: str | Path | None = None,
        table_offset: int = 12,
        header_padding: bytes = b"",
    ) -> None:
        values = [-1] * 40 if stringrefs is None else [int(value) for value in stringrefs]
        if len(values) < 28:
            raise ValueError("A KOTOR SSF document must contain at least the 28 named sound slots.")
        if int(table_offset) < 12:
            raise ValueError("An SSF sound-table offset cannot overlap its 12-byte header.")
        padding = bytes(header_padding)
        if len(padding) != int(table_offset) - 12:
            raise ValueError("SSF header padding must exactly fill the declared table offset.")
        self._stringrefs = values
        self.source_path = Path(source_path) if source_path is not None else None
        self.table_offset = int(table_offset)
        self.header_padding = padding

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "SoundSetDocument":
        data, source_path = _read_source(source)
        if len(data) < 12:
            raise ValueError("SSF data is shorter than its 12-byte V1.1 header.")
        if data[:4] != b"SSF " or data[4:8] != b"V1.1":
            raise ValueError("Expected an SSF V1.1 resource.")
        table_offset = struct.unpack_from("<I", data, 8)[0]
        if table_offset < 12 or table_offset > len(data):
            raise ValueError("SSF sound-table offset is outside the resource.")
        table_size = len(data) - table_offset
        if table_size % 4:
            raise ValueError("SSF sound-table byte count is not divisible by four.")
        entry_count = table_size // 4
        if entry_count < 28:
            raise ValueError("SSF sound table does not contain all 28 named slots.")
        raw_values = struct.unpack_from(f"<{entry_count}I", data, table_offset)
        values = [-1 if value == 0xFFFFFFFF else int(value) for value in raw_values]

        # PyKotor independently checks the named portion used by the public
        # engine model.  The bounded parser above remains authoritative for the
        # retail tail because PyKotor intentionally exposes only 28 names.
        from pykotor.resource.formats.ssf import read_ssf

        read_ssf(data)
        return cls(
            values,
            source_path=source_path,
            table_offset=table_offset,
            header_padding=data[12:table_offset],
        )

    @staticmethod
    def slot_names() -> tuple[str, ...]:
        from pykotor.resource.formats.ssf import SSFSound

        return tuple(slot.name for slot in SSFSound)

    @property
    def stringrefs(self) -> tuple[int, ...]:
        return tuple(self._stringrefs)

    @property
    def named_stringrefs(self) -> tuple[int, ...]:
        return tuple(self._stringrefs[:28])

    @property
    def unknown_entries(self) -> tuple[tuple[int, int], ...]:
        return tuple(enumerate(self._stringrefs[28:], start=28))

    def snapshot(self) -> SoundSetSnapshot:
        return SoundSetSnapshot(self.stringrefs, self.table_offset, self.header_padding)

    def restore(self, snapshot: SoundSetSnapshot) -> None:
        if len(snapshot.stringrefs) < 28:
            raise ValueError("A KOTOR SSF snapshot must retain at least 28 sound slots.")
        self._stringrefs = list(snapshot.stringrefs)
        self.table_offset = int(snapshot.table_offset)
        self.header_padding = bytes(snapshot.header_padding)

    @staticmethod
    def _slot_index(slot: int | str) -> int:
        from pykotor.resource.formats.ssf import SSFSound

        if isinstance(slot, str):
            try:
                return int(SSFSound[slot.strip().upper()].value)
            except KeyError as exc:
                raise KeyError(f"Unknown SSF sound slot: {slot}") from exc
        index = int(slot)
        if not 0 <= index < 28:
            raise IndexError(f"SSF sound slot is out of range: {index}")
        return index

    def get_slot(self, slot: int | str) -> int:
        return self._stringrefs[self._slot_index(slot)]

    def set_slot(self, slot: int | str, stringref: int) -> int:
        index = self._slot_index(slot)
        value = int(stringref)
        if value < -1 or value > 0xFFFFFFFE:
            raise ValueError("An SSF StrRef must be -1 (unset) or at most 0xFFFFFFFE.")
        previous = self._stringrefs[index]
        self._stringrefs[index] = value
        return previous

    def set_unknown_entry(self, index: int, stringref: int) -> int:
        """Edit an explicitly stored unnamed retail entry without truncation."""

        position = int(index)
        if position < 28 or position >= len(self._stringrefs):
            raise IndexError("An unnamed SSF entry index must refer to an existing entry after slot 27.")
        value = int(stringref)
        if value < -1 or value > 0xFFFFFFFE:
            raise ValueError("An SSF StrRef must be -1 (unset) or at most 0xFFFFFFFE.")
        previous = self._stringrefs[position]
        self._stringrefs[position] = value
        return previous

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics: list[DataDiagnostic] = []
        if len(self._stringrefs) < 28:
            diagnostics.append(DataDiagnostic("blocking", "ssf.slot_count", "SSF resources require at least 28 named sound slots."))
        names = self.slot_names()
        for index, stringref in enumerate(self._stringrefs):
            if stringref < -1 or stringref > 0xFFFFFFFE:
                diagnostics.append(
                    DataDiagnostic(
                        "blocking",
                        "ssf.strref_range",
                        "SSF StrRefs must be -1 (unset) or at most 0xFFFFFFFE.",
                        names[index] if index < len(names) else str(index),
                    )
                )
        return tuple(diagnostics)

    def to_bytes(self) -> bytes:
        blocking = [diagnostic.message for diagnostic in self.validate() if diagnostic.blocking]
        if blocking:
            raise ValueError("Cannot serialize invalid SSF: " + "; ".join(blocking))
        raw_values = [0xFFFFFFFF if value == -1 else int(value) for value in self._stringrefs]
        output = (
            b"SSF "
            + b"V1.1"
            + struct.pack("<I", self.table_offset)
            + self.header_padding
            + struct.pack(f"<{len(raw_values)}I", *raw_values)
        )
        from pykotor.resource.formats.ssf import read_ssf

        read_ssf(output)
        return output

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError("A target path is required for a new SSF document.")
        result = _atomic_write(target, self.to_bytes())
        self.source_path = result
        return result


# ---------------------------------------------------------------------------
# LIP animation keyframes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class LipKeyframeRecord:
    time: float
    shape: int


@dataclass(frozen=True)
class LipSnapshot:
    duration: float
    keyframes: tuple[LipKeyframeRecord, ...]


class LipDocument:
    """Editable KOTOR LIP V1.0 viseme timeline."""

    def __init__(
        self,
        duration: float = 0.0,
        keyframes: Sequence[LipKeyframeRecord] = (),
        *,
        source_path: str | Path | None = None,
    ) -> None:
        self.duration = float(duration)
        self._keyframes = sorted((LipKeyframeRecord(float(frame.time), int(frame.shape)) for frame in keyframes), key=lambda frame: frame.time)
        self.source_path = Path(source_path) if source_path is not None else None

    @classmethod
    def load(cls, source: bytes | bytearray | memoryview | str | Path) -> "LipDocument":
        from pykotor.resource.formats.lip import read_lip

        data, source_path = _read_source(source)
        lip = read_lip(data)
        frames = [LipKeyframeRecord(float(frame.time), int(frame.shape.value)) for frame in lip.frames]
        return cls(float(lip.length), frames, source_path=source_path)

    @property
    def keyframes(self) -> tuple[LipKeyframeRecord, ...]:
        return tuple(self._keyframes)

    @staticmethod
    def shape_names() -> tuple[str, ...]:
        from pykotor.resource.formats.lip import LIPShape

        return tuple(shape.name for shape in LIPShape)

    @staticmethod
    def shape_for_phoneme(phoneme: str) -> int:
        from pykotor.resource.formats.lip import LIPShape

        return int(LIPShape.from_phoneme(str(phoneme)).value)

    def snapshot(self) -> LipSnapshot:
        return LipSnapshot(self.duration, self.keyframes)

    def restore(self, snapshot: LipSnapshot) -> None:
        self.duration = float(snapshot.duration)
        self._keyframes = list(snapshot.keyframes)

    def set_duration(self, duration: float) -> None:
        value = float(duration)
        if value < 0:
            raise ValueError("LIP duration cannot be negative.")
        if self._keyframes and value < self._keyframes[-1].time:
            raise ValueError("LIP duration cannot end before its last keyframe.")
        self.duration = value

    def add_keyframe(self, time: float, shape: int | str) -> int:
        from pykotor.resource.formats.lip import LIPShape

        time_value = float(time)
        if time_value < 0:
            raise ValueError("LIP keyframe time cannot be negative.")
        if isinstance(shape, str):
            try:
                shape_value = int(LIPShape[shape.strip().upper()].value)
            except KeyError as exc:
                raise ValueError(f"Unknown LIP shape: {shape}") from exc
        else:
            shape_value = int(shape)
            try:
                LIPShape(shape_value)
            except ValueError as exc:
                raise ValueError("LIP shape index must be from 0 to 15.") from exc
        self._keyframes = [frame for frame in self._keyframes if abs(frame.time - time_value) > 0.0001]
        self._keyframes.append(LipKeyframeRecord(time_value, shape_value))
        self._keyframes.sort(key=lambda frame: frame.time)
        self.duration = max(self.duration, time_value)
        return self._keyframes.index(LipKeyframeRecord(time_value, shape_value))

    def update_keyframe(self, index: int, *, time: float | None = None, shape: int | str | None = None) -> int:
        current = self._keyframes[index]
        self._keyframes.pop(index)
        try:
            return self.add_keyframe(current.time if time is None else time, current.shape if shape is None else shape)
        except Exception:
            self._keyframes.insert(index, current)
            self._keyframes.sort(key=lambda frame: frame.time)
            raise

    def remove_keyframe(self, index: int) -> LipKeyframeRecord:
        return self._keyframes.pop(index)

    def validate(self) -> tuple[DataDiagnostic, ...]:
        diagnostics: list[DataDiagnostic] = []
        if self.duration < 0:
            diagnostics.append(DataDiagnostic("blocking", "lip.negative_duration", "LIP duration cannot be negative."))
        previous = -1.0
        seen_times: set[float] = set()
        for frame in self._keyframes:
            if frame.time < 0:
                diagnostics.append(DataDiagnostic("blocking", "lip.negative_time", "LIP keyframe times cannot be negative.", str(frame.time)))
            if frame.time < previous:
                diagnostics.append(DataDiagnostic("blocking", "lip.order", "LIP keyframes must be in ascending order."))
            if frame.time in seen_times:
                diagnostics.append(DataDiagnostic("blocking", "lip.duplicate_time", "LIP keyframe times must be unique.", str(frame.time)))
            if not 0 <= frame.shape <= 15:
                diagnostics.append(DataDiagnostic("blocking", "lip.shape", "LIP shape indices must be from 0 to 15.", str(frame.shape)))
            if frame.time > self.duration:
                diagnostics.append(DataDiagnostic("blocking", "lip.duration", "A LIP keyframe falls after the animation duration.", str(frame.time)))
            previous = frame.time
            seen_times.add(frame.time)
        return tuple(diagnostics)

    def to_bytes(self) -> bytes:
        from pykotor.resource.formats.lip import LIP, LIPKeyFrame, LIPShape, bytes_lip, read_lip

        blocking = [diagnostic.message for diagnostic in self.validate() if diagnostic.blocking]
        if blocking:
            raise ValueError("Cannot serialize invalid LIP: " + "; ".join(blocking))
        lip = LIP()
        lip.length = float(self.duration)
        lip.frames = [LIPKeyFrame(float(frame.time), LIPShape(int(frame.shape))) for frame in self._keyframes]
        output = bytes(bytes_lip(lip))
        read_lip(output)
        return output

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError("A target path is required for a new LIP document.")
        result = _atomic_write(target, self.to_bytes())
        self.source_path = result
        return result


class NarrativeDataAuthoringService:
    """Small facade used by product controllers without importing Qt."""

    @staticmethod
    def open_2da(source: bytes | bytearray | memoryview | str | Path) -> TwoDADocument:
        return TwoDADocument.load(source)

    @staticmethod
    def open_tlk(source: bytes | bytearray | memoryview | str | Path) -> TalkTableDocument:
        return TalkTableDocument.load(source)

    @staticmethod
    def open_jrl(source: bytes | bytearray | memoryview | str | Path) -> JournalDocument:
        return JournalDocument.load(source)

    @staticmethod
    def open_globals(source: bytes | bytearray | memoryview | str | Path) -> GlobalVariableTable:
        return GlobalVariableTable.load(source)

    @staticmethod
    def open_ssf(source: bytes | bytearray | memoryview | str | Path) -> SoundSetDocument:
        return SoundSetDocument.load(source)

    @staticmethod
    def open_lip(source: bytes | bytearray | memoryview | str | Path) -> LipDocument:
        return LipDocument.load(source)


__all__ = [
    "DataDiagnostic",
    "GLOBAL_BOOLEAN",
    "GLOBAL_LOCATION",
    "GLOBAL_NUMBER",
    "GLOBAL_STRING",
    "GlobalVariableRecord",
    "GlobalVariableTable",
    "JournalDocument",
    "JournalEntryRecord",
    "JournalQuestRecord",
    "JournalSnapshot",
    "LipDocument",
    "LipKeyframeRecord",
    "LipSnapshot",
    "LocalizedText",
    "NarrativeDataAuthoringService",
    "SoundSetDocument",
    "SoundSetSnapshot",
    "TalkTableDocument",
    "TalkTableEntry",
    "TalkTableSnapshot",
    "TwoDADocument",
    "TwoDASnapshot",
]
