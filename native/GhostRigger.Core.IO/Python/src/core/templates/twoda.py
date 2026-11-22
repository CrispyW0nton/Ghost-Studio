"""
KotOR 2DA File Reader
Supports both formats:
  - Binary "2DA V2.b" (packed inside KEY/BIF archives)
  - ASCII  "2DA V2.0" (loose .2da files in Override or tools)

Usage:
    tda = TwoDA.from_bytes(raw_bytes)          # auto-detect format
    tda = TwoDA.from_file("appearance.2da")    # load from disk

    # Access
    print(tda.columns)                          # list of column names
    print(len(tda))                             # number of rows
    val = tda.get(row_idx, "label")             # get cell by row + column name
    val = tda[row_idx]["label"]                 # same via subscript
    rows = tda.find("model", "c_bantha")        # find rows where col=value
    tda.to_tsv("/tmp/appearance.tsv")           # export as TSV
"""

import struct
import logging
import ctypes
import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

from src.core.templates._native import native_templates

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 2DA row: lightweight dict-like wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TwoDARow:
    __slots__ = ('_idx', '_cols', '_data', '_label')

    def __init__(self, idx: int, columns: List[str], data: List[str], label: Optional[str] = None):
        self._idx  = idx
        self._cols = columns
        self._data = data
        self._label = str(label) if label is not None else str(idx)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def label(self) -> str:
        return self._label

    def __getitem__(self, key) -> str:
        if isinstance(key, int):
            return self._data[key] if key < len(self._data) else ''
        key_l = key.lower()
        for i, c in enumerate(self._cols):
            if c.lower() == key_l:
                return self._data[i] if i < len(self._data) else ''
        return ''

    def __contains__(self, key: str) -> bool:
        return key.lower() in [c.lower() for c in self._cols]

    def get(self, key: str, default: str = '') -> str:
        v = self[key]
        return _twoda_cell_or_default(v, default)

    def as_dict(self) -> Dict[str, str]:
        return {c: (self._data[i] if i < len(self._data) else '')
                for i, c in enumerate(self._cols)}

    def __repr__(self) -> str:
        return f"TwoDARow({self._idx}: {self.as_dict()})"


# ─────────────────────────────────────────────────────────────────────────────
# Main 2DA class
# ─────────────────────────────────────────────────────────────────────────────

class TwoDA:
    """
    Parsed KotOR 2DA table.
    Supports both binary V2.b and ASCII V2.0 formats.
    """

    # sentinel for empty cells
    BLANK = '****'

    def __init__(self, name: str = ''):
        self.name:    str         = name
        self.columns: List[str]   = []
        self._rows:   List[List[str]] = []   # list of raw cell value lists
        self._labels: List[str]   = []

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_bytes(cls, data: bytes, name: str = '') -> 'TwoDA':
        """Auto-detect binary or ASCII format and parse."""
        if not data:
            raise ValueError("Empty data")
        pykotor_table = cls._parse_with_pykotor(data, name)
        if pykotor_table is not None:
            return pykotor_table
        native_format = _detect_twoda_format(data)
        if native_format == "binary_v2b":
            return cls._parse_binary(data, name)
        if native_format == "ascii_v2":
            return cls._parse_ascii(data, name)
        # Peek at magic
        header = data[:10]
        if header[:3] == b'2DA':
            ver = header[4:9]
            if ver == b'V2.b\n':
                return cls._parse_binary(data, name)
            elif ver.startswith(b'V2.'):
                return cls._parse_ascii(data, name)
        raise ValueError(f"Unknown 2DA format: {data[:16]!r}")

    @classmethod
    def _parse_with_pykotor(cls, data: bytes, name: str) -> Optional['TwoDA']:
        """Parse through PyKotor when available, then normalize to this API."""
        try:
            from pykotor.resource.formats.twoda import read_2da

            parsed = read_2da(data)
        except Exception:
            return None

        tda = cls(name)
        try:
            tda.columns = [str(column) for column in parsed.get_headers()]
        except Exception:
            return None

        rows: List[List[str]] = []
        labels: List[str] = []
        for row in parsed:
            label = getattr(row, "label", "")
            labels.append(str(label() if callable(label) else label))
            values: List[str] = []
            for column in tda.columns:
                try:
                    values.append(str(row.get_string(column) or ""))
                except Exception:
                    values.append("")
            rows.append(values)
        tda._rows = rows
        tda._labels = labels
        return tda

    @classmethod
    def from_file(cls, path: str) -> 'TwoDA':
        """Load from a loose .2da file on disk."""
        import os
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            data = f.read()
        return cls.from_bytes(data, name)

    # ── Binary V2.b parser ────────────────────────────────────────────────────

    @classmethod
    def _parse_binary(cls, data: bytes, name: str) -> 'TwoDA':
        """
        KotOR binary 2DA format (V2.b) – confirmed layout from game files:

          Offset  0-8:  "2DA V2.b\\n"              (9 bytes, header)
          Offset  9:    Column-name block           (tab-separated ASCII, \\0-terminated)
          After cols:   uint32 row count            (4 bytes, little-endian)
          After count:  Row-label block             (tab-separated ASCII, \\0-terminated)
          After labels: uint16 cell-offset table    (row_count × col_count entries)
          After table:  String data block           (null-terminated strings, offset=0 means empty)
        """
        tda = cls(name)

        if len(data) < 14:
            return tda

        pos = 9  # after "2DA V2.b\n"

        # ── 1. Column-name block (tab-separated, single \0-terminated) ─────
        col_block_end = data.find(b'\x00', pos)
        if col_block_end < 0:
            return tda
        col_block = data[pos:col_block_end].decode('latin-1', errors='replace')
        tda.columns = [c for c in col_block.split('\t') if c]
        pos = col_block_end + 1

        n_cols = len(tda.columns)
        if n_cols == 0:
            return tda

        # ── 2. uint32 row count ──────────────────────────────────────────────
        if pos + 4 > len(data):
            return tda
        n_rows = struct.unpack_from('<I', data, pos)[0]
        pos += 4

        # ── 3. Row-label block (tab-separated ASCII, \0-terminated) ─────────
        row_block_end = data.find(b'\x00', pos)
        if row_block_end < 0:
            return tda
        row_block = data[pos:row_block_end].decode('latin-1', errors='replace')
        tda._labels = [label for label in row_block.split('\t') if label][:n_rows]
        # Preserve labels because some game 2DAs use them as durable row ids.
        pos = row_block_end + 1

        # ── 4. uint16 cell-offset table (n_rows × n_cols entries) ───────────
        n_cells = n_rows * n_cols
        cell_table_size = n_cells * 2
        if pos + cell_table_size > len(data):
            # Recover partial data
            available_cells = (len(data) - pos) // 2
            n_rows = (min(n_rows,
                          available_cells // n_cols if n_cols else 0))
            n_cells = n_rows * n_cols
            cell_table_size = n_cells * 2

        cell_offsets = struct.unpack_from(f'<{n_cells}H', data, pos)
        pos += cell_table_size

        # ── 5. String data block ─────────────────────────────────────────────
        str_data = data[pos:]

        def get_str(offset: int) -> str:
            if offset >= len(str_data):
                return ''
            end = str_data.find(b'\x00', offset)
            if end < 0:
                end = len(str_data)
            raw_s = str_data[offset:end]
            return raw_s.decode('latin-1', errors='replace')

        tda._rows = []
        while len(tda._labels) < n_rows:
            tda._labels.append(str(len(tda._labels)))
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                cell_idx = r * n_cols + c
                if cell_idx < len(cell_offsets):
                    row.append(get_str(cell_offsets[cell_idx]))
                else:
                    row.append('')
            tda._rows.append(row)

        return tda

    # ── ASCII V2.0 parser ─────────────────────────────────────────────────────

    @classmethod
    def _parse_ascii(cls, data: bytes, name: str) -> 'TwoDA':
        """
        ASCII 2DA format:
          Line 0: "2DA V2.0"
          Line 1: empty (blank row label line)
          Line 2: column header line (whitespace-separated)
          Lines 3+: data rows (first token is row index, then values)
        """
        tda = cls(name)

        text = data.decode('latin-1', errors='replace')
        lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

        if len(lines) < 3:
            return tda

        # Line 2 = column headers
        header_line = lines[2] if len(lines) > 2 else ''
        tda.columns = _split_2da_line(header_line)

        n_cols = len(tda.columns)
        tda._rows = []
        tda._labels = []

        for line in lines[3:]:
            line = line.strip()
            if not line:
                continue
            tokens = _split_2da_line(line)
            if not tokens:
                continue
            # First token is row index (integer label), skip it
            tda._labels.append(tokens[0])
            row_data = tokens[1:] if len(tokens) > 1 else []
            # Pad / trim to n_cols
            while len(row_data) < n_cols:
                row_data.append('')
            tda._rows.append(row_data[:n_cols])

        return tda

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._rows)

    def _row_label(self, idx: int) -> str:
        if idx < len(self._labels):
            return self._labels[idx]
        return str(idx)

    def __iter__(self) -> Iterator[TwoDARow]:
        for i, row_data in enumerate(self._rows):
            yield TwoDARow(i, self.columns, row_data, self._row_label(i))

    def __getitem__(self, idx: int) -> TwoDARow:
        return TwoDARow(idx, self.columns, self._rows[idx], self._row_label(idx))

    def get(self, row: int, col: str, default: str = '') -> str:
        """Get cell value at (row, col_name). Returns default if missing/blank."""
        if row < 0 or row >= len(self._rows):
            return default
        row_data = self._rows[row]
        col_l = col.lower()
        for i, c in enumerate(self.columns):
            if c.lower() == col_l:
                v = row_data[i] if i < len(row_data) else ''
                return _twoda_cell_or_default(v, default)
        return default

    def get_int(self, row: int, col: str, default: int = -1) -> int:
        """Get cell as integer."""
        v = self.get(row, col)
        try:
            return int(v)
        except (ValueError, TypeError):
            return default

    def get_float(self, row: int, col: str, default: float = 0.0) -> float:
        """Get cell as float."""
        v = self.get(row, col)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    def col_index(self, col_name: str) -> int:
        """Return index of column, or -1 if not found."""
        col_l = col_name.lower()
        for i, c in enumerate(self.columns):
            if c.lower() == col_l:
                return i
        return -1

    def find(self, col: str, value: str,
             case_sensitive: bool = False) -> List[TwoDARow]:
        """Find all rows where column == value."""
        ci = self.col_index(col)
        if ci < 0:
            return []
        results = []
        v_cmp = value if case_sensitive else value.lower()
        for i, row_data in enumerate(self._rows):
            cell = row_data[ci] if ci < len(row_data) else ''
            cell_cmp = cell if case_sensitive else cell.lower()
            if cell_cmp == v_cmp:
                results.append(TwoDARow(i, self.columns, row_data, self._row_label(i)))
        return results

    def find_first(self, col: str, value: str) -> Optional[TwoDARow]:
        """Return first matching row or None."""
        rows = self.find(col, value)
        return rows[0] if rows else None

    def column_values(self, col: str) -> List[str]:
        """Return all values in a column (empty/blank replaced with '')."""
        ci = self.col_index(col)
        if ci < 0:
            return []
        return [(row_data[ci] if ci < len(row_data) else '')
                for row_data in self._rows]

    # ── Export ────────────────────────────────────────────────────────────────

    def to_tsv(self, path: str):
        """Write as TSV (tab-separated values), row 0 = header."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\t'.join(['#'] + self.columns) + '\n')
            for i, row_data in enumerate(self._rows):
                cells = [self._row_label(i)] + [(c if c else '') for c in row_data]
                f.write('\t'.join(cells) + '\n')

    def to_ascii_2da(self) -> str:
        """Produce ASCII 2DA V2.0 text (suitable for Override folder)."""
        lines = ['2DA V2.0', '']
        # Column header line
        lines.append('          ' + '  '.join(c.ljust(16) for c in self.columns))
        for i, row_data in enumerate(self._rows):
            cells = [(c if c else self.BLANK) for c in row_data]
            row_str = f"{self._row_label(i):<10}" + '  '.join(c.ljust(16) for c in cells)
            lines.append(row_str)
        return '\n'.join(lines) + '\n'

    def __repr__(self) -> str:
        return (f"TwoDA({self.name!r}: {len(self.columns)} cols × "
                f"{len(self._rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_twoda_format(data: bytes) -> str:
    dll = native_templates()
    if dll is not None:
        try:
            payload = bytes(data or b"")
            array = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
            raw = dll.gr_templates_detect_twoda_format(array, len(payload))
            if raw:
                return raw.decode("utf-8")
        except (OSError, ValueError):
            pass
    return _python_detect_twoda_format(data)


def _python_detect_twoda_format(data: bytes) -> str:
    if not data:
        return "empty"
    header = data[:10]
    if header[:3] == b'2DA':
        ver = header[4:9]
        if ver == b'V2.b\n':
            return "binary_v2b"
        if ver.startswith(b'V2.'):
            return "ascii_v2"
    return "unknown"


def _twoda_cell_or_default(value: str, fallback: str) -> str:
    dll = native_templates()
    if dll is not None:
        try:
            raw = dll.gr_templates_twoda_cell_or_default(
                str(value or "").encode("utf-8"),
                str(fallback or "").encode("utf-8"),
            )
            if raw is not None:
                return raw.decode("utf-8")
        except OSError:
            pass
    return _python_twoda_cell_or_default(value, fallback)


def _python_twoda_cell_or_default(value: str, fallback: str) -> str:
    return value if value not in ('', '****') else fallback


def _python_split_2da_line(line: str) -> List[str]:
    """
    Split a 2DA ASCII data line, respecting quoted strings.
    Empty tokens (****) are kept as empty string.
    """
    tokens: List[str] = []
    current: List[str] = []
    in_quote = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"':
            in_quote = not in_quote
        elif c in (' ', '\t') and not in_quote:
            if current:
                tok = ''.join(current)
                tokens.append('' if tok == '****' else tok)
                current = []
        else:
            current.append(c)
        i += 1
    if current:
        tok = ''.join(current)
        tokens.append('' if tok == '****' else tok)
    return tokens


def _split_2da_line(line: str) -> List[str]:
    dll = native_templates()
    if dll is not None:
        try:
            raw = dll.gr_templates_split_twoda_line_json(str(line or "").encode("utf-8"))
            if raw:
                values = json.loads(raw.decode("utf-8"))
                if isinstance(values, list) and all(isinstance(value, str) for value in values):
                    return values
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return _python_split_2da_line(line)


# ─────────────────────────────────────────────────────────────────────────────
# 2DA Cache — load-on-demand from GameLibrary
# ─────────────────────────────────────────────────────────────────────────────

class TwoDACache:
    """
    Load-on-demand cache for all game 2DA files.
    Reads from KEY/BIF via GameLibrary.

    Usage:
        cache = TwoDACache(game_library)
        appearance = cache.get("appearance")
        print(appearance.get(3, "modela"))
    """

    def __init__(self, game_library=None):
        self._library = game_library
        self._cache: Dict[str, TwoDA] = {}

    def set_library(self, lib):
        self._library = lib
        self._cache.clear()

    def get(self, name: str, game: str = "K1") -> Optional[TwoDA]:
        """Get 2DA by name (case-insensitive). Cached after first load."""
        key = name.lower()
        if key in self._cache:
            return self._cache[key]

        raw = self._fetch_raw(key, game)
        if raw is None:
            log.debug(f"2DA not found: {name}")
            return None
        try:
            tda = TwoDA.from_bytes(raw, name=key)
            self._cache[key] = tda
            return tda
        except Exception as e:
            log.warning(f"Failed to parse 2DA {name!r}: {e}")
            return None

    def _fetch_raw(self, name: str, game: str) -> Optional[bytes]:
        """Fetch raw bytes for a 2DA resource."""
        if self._library is None:
            return None

        from ..game.game_library_ext import RES_2DA  # avoid circular import
        reader = (self._library._k1_key if game == "K1"
                  else self._library._k2_key)
        if reader:
            e = reader.get(name, RES_2DA)
            if e:
                try:
                    return e.read()
                except Exception as ex:
                    log.debug(f"2DA read error {name}: {ex}")

        # Also check Override folder
        if self._library:
            gdir = (self._library.k1_dir if game == "K1"
                    else self._library.k2_dir)
            if gdir:
                from pathlib import Path
                override = Path(gdir) / 'Override'
                for p in [override / f"{name}.2da",
                          override / f"{name.upper()}.2DA"]:
                    if p.exists():
                        return p.read_bytes()
        return None

    def list_all(self, game: str = "K1") -> List[str]:
        """List all available 2DA names."""
        if self._library is None:
            return []
        from ..game.game_library_ext import RES_2DA
        reader = (self._library._k1_key if game == "K1"
                  else self._library._k2_key)
        if reader:
            return sorted(e.resref.lower()
                          for e in reader.list_type(RES_2DA))
        return []

    def preload_all(self, game: str = "K1", progress_cb=None):
        """Preload all 2DA files into the cache."""
        names = self.list_all(game)
        total = len(names)
        for i, name in enumerate(names):
            if progress_cb and i % 20 == 0:
                progress_cb(f"Loading 2DAs: {i}/{total} ({name})")
            self.get(name, game)
        if progress_cb:
            progress_cb(f"Loaded {total} 2DA files")

    def clear(self):
        self._cache.clear()

    def __repr__(self) -> str:
        return f"TwoDACache({len(self._cache)} cached)"
