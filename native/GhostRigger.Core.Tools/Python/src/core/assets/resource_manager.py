"""
resource_manager.py — Unified KotOR Resource Manager
=====================================================
Single source of truth for all KotOR 1 & 2 resource access.

Architecture
------------
Rebuilt from ground up taking the best ideas from:
  • KotorBlender: header-only ERF/BIF parsing, lazy seek-on-demand reads
  • PyKotor:     Installation abstraction, module ERF support, override priority

Design principles
-----------------
1. Index-only init (< 150 ms): reads only file headers and key tables,
   never loads actual resource data during startup.
2. Lazy seek reads (< 2 ms): each get() does a single lseek + read.
3. Priority chain (matches KotOR engine):
      Override/ > module ERF > TexturePacks ERF > streamed audio > BIF
4. Module support: any .mod/.rim/.erf in modules/ is auto-indexed.
5. Single unified API: one get(name, type) call handles everything.
6. Thread-safe: immutable install indexes remain lock-free internally; manager
   install/overlay publications are revisioned under an RLock, and file reads
   use per-call open() for safety.

Resource type constants (NWN / KotOR standard)
----------------------------------------------
MDL  = 2002   .mdl  binary model geometry
MDX  = 3008   .mdx  binary model vertex data
TPC  = 3007   .tpc  TPC compressed texture
TGA  = 3       .tga  TGA/TPC texture (KotOR stores TPC with .tga extension)
TXI  = 2014   .txi  texture parameters
UTC  = 2023   .utc  creature template
ARE  = 2012   .are  area data
IFO  = 2013   .ifo  module info
DLG  = 2029   .dlg  dialog tree
WOK  = 2016   .wok  room walkmesh (BWM)
LYT  = 3000   .lyt  area layout
VIS  = 3001   .vis  area visibility
PTH  = 3003   .pth  area path graph
TwoDA  = 2017   .2da  two-dimensional array
GIT  = 2015   .git  area instance template
NSS  = 2009   .nss  NWScript source
NCS  = 2010   .ncs  NWScript compiled
SSF  = 2015   .ssf  sound set file
WAV  = 4       .wav  streamed audio
MP3  = 25014   .mp3  streamed audio
"""

from __future__ import annotations

import os
import struct
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..game.pykotor_gff_runtime_fix import ensure_pykotor_gff_acquire_quiet

log = logging.getLogger(__name__)

# ── Resource type constants ──────────────────────────────────────────────────

RES_BMP  = 1
RES_TGA  = 3
RES_WAV  = 4
RES_MP3  = 25014
RES_PLT  = 6
RES_INI  = 7
RES_TXT  = 10
RES_MDL  = 2002
RES_NSS  = 2009
RES_NCS  = 2010
RES_MDX  = 3008
RES_TXI  = 2014
RES_ARE  = 2012
RES_IFO  = 2013
# Blueprint (UT*) resource type IDs — canonical Aurora/KOTOR values, matching
# chitin.key and PyKotor's ResourceType. (Earlier RES_UTC=2023/RES_UTD=2038
# were wrong, which silently broke creature/door model resolution for imported
# modules — their bytes were indexed under a type id nothing ever requested.)
RES_UTC  = 2027   # creature
RES_UTI  = 2025   # item
RES_UTT  = 2032   # trigger
RES_UTS  = 2035   # sound
RES_UTE  = 2040   # encounter
RES_UTD  = 2042   # door
RES_UTP  = 2044   # placeable
RES_UTM  = 2051   # store / merchant
RES_UTW  = 2058   # waypoint
RES_DLG  = 2029
RES_TPC  = 3007
RES_LYT  = 3000
RES_VIS  = 3001
RES_PTH  = 3003
RES_2DA  = 2017
RES_GIT  = 2015
RES_MOD  = 3011  # module reference
RES_WOK  = 2016  # WOK/BWM room walkmesh (confirmed from KEY/RIM resources)

# Extension ↔ type tables
EXT_TO_TYPE: Dict[str, int] = {
    'mdl': RES_MDL, 'mdx': RES_MDX,
    'tpc': RES_TPC, 'tga': RES_TGA, 'txi': RES_TXI,
    'utc': RES_UTC, 'utp': RES_UTP, 'utd': RES_UTD, 'utm': RES_UTM,
    'uti': RES_UTI, 'utt': RES_UTT, 'uts': RES_UTS, 'ute': RES_UTE, 'utw': RES_UTW,
    'are': RES_ARE, 'ifo': RES_IFO, 'dlg': RES_DLG,
    'lyt': RES_LYT, 'vis': RES_VIS, 'pth': RES_PTH, '2da': RES_2DA,
    'git': RES_GIT, 'wok': RES_WOK,
    'wav': RES_WAV, 'mp3': RES_MP3,
    'bmp': RES_BMP, 'ini': RES_INI, 'txt': RES_TXT,
    'nss': RES_NSS, 'ncs': RES_NCS,
}
TYPE_TO_EXT: Dict[int, str] = {v: k for k, v in EXT_TO_TYPE.items()}

_LOOSE_AUDIO_EXT_TYPES: Dict[str, int] = {
    '.wav': RES_WAV,
    '.mp3': RES_MP3,
}


def _key(name: str, res_type: int) -> str:
    """Canonical lookup key: lowercase name + type."""
    return f"{name.lower()}:{res_type}"


_TEXTURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    # TSL's c_drexlf.mdl references c_drex01, but the shipped texture pack
    # stores the diffuse as c_drexl01.
    'c_drex01': ('c_drexl01',),
    # The common HQ-blaster override patches the assassin/war droid models to
    # request x_rptnblstr_001. A partial install can leave only its TXI while
    # K2's canonical repeating-blaster diffuse remains available under w_.
    'x_rptnblstr_001': ('w_rptnblstr_001',),
}


def _texture_name_candidates(name: str) -> Tuple[str, ...]:
    """Return texture lookup candidates, preserving the authored name first."""
    key = (name or '').strip().lower()
    if not key:
        return ()

    candidates = [key]
    for alias in _TEXTURE_ALIASES.get(key, ()):
        alias_key = alias.strip().lower()
        if alias_key and alias_key not in candidates:
            candidates.append(alias_key)
    return tuple(candidates)


# ── Low-level archive readers ────────────────────────────────────────────────

class _BifIndex:
    """
    Reads only the variable-resource table from a BIFF V1 file.
    Table: entry_count × 16 bytes (ID[4], Offset[4], FileSize[4], ResType[4]).
    Actual data is read on demand via seek+read.
    """
    __slots__ = ('path', '_table')  # _table: dict[var_idx → (offset, size)]

    def __init__(self, path: str):
        self.path = path
        self._table: Dict[int, Tuple[int, int]] = {}
        try:
            with open(path, 'rb') as fh:
                hdr = fh.read(20)
                var_count = struct.unpack_from('<I', hdr, 8)[0]
                raw = fh.read(var_count * 16)
            for i in range(var_count):
                b = i * 16
                offset   = struct.unpack_from('<I', raw, b + 4)[0]
                filesize = struct.unpack_from('<I', raw, b + 8)[0]
                self._table[i] = (offset, filesize)
        except Exception as exc:
            log.warning(f"BIF index failed {path}: {exc}")

    def read(self, var_idx: int) -> Optional[bytes]:
        slot = self._table.get(var_idx)
        if slot is None:
            return None
        offset, size = slot
        try:
            with open(self.path, 'rb') as fh:
                fh.seek(offset)
                return fh.read(size)
        except Exception as exc:
            log.warning(f"BIF read error {self.path}[{var_idx}]: {exc}")
            return None


class _ErfIndex:
    """
    Reads only the key + resource lists from an ERF/MOD/RIM/SAV file.
    ERF V1 layout:
      Header  160 bytes
        [16]: entry_count  uint32
        [24]: off_keys     uint32  (key list offset)
        [28]: off_res      uint32  (resource list offset)
      Key list:  entry_count × 24 bytes
        resref[16]  resID[4]  resType[2]  unused[2]
      Resource list:  entry_count × 8 bytes
        offset[4]  size[4]
    No resource data is read at index time.
    """
    __slots__ = ('path', '_index')  # _index: dict[key → (offset, size)]

    def __init__(self, path: str):
        self.path = path
        self._index: Dict[str, Tuple[int, int]] = {}
        try:
            with open(path, 'rb') as fh:
                hdr = fh.read(160)
            entry_count = struct.unpack_from('<I', hdr, 16)[0]
            off_keys    = struct.unpack_from('<I', hdr, 24)[0]
            off_res     = struct.unpack_from('<I', hdr, 28)[0]
            if entry_count == 0:
                return
            with open(path, 'rb') as fh:
                fh.seek(off_keys)
                key_raw = fh.read(entry_count * 24)
                fh.seek(off_res)
                res_raw = fh.read(entry_count * 8)
            for i in range(entry_count):
                kb = i * 24
                rb = i * 8
                resref   = key_raw[kb:kb+16].split(b'\x00', 1)[0].decode('ascii', 'replace').lower()
                res_type = struct.unpack_from('<H', key_raw, kb + 20)[0]
                offset   = struct.unpack_from('<I', res_raw, rb)[0]
                size     = struct.unpack_from('<I', res_raw, rb + 4)[0]
                k = _key(resref, res_type)
                self._index[k] = (offset, size)
        except Exception as exc:
            log.warning(f"ERF index failed {path}: {exc}")

    def read(self, name: str, res_type: int) -> Optional[bytes]:
        slot = self._index.get(_key(name, res_type))
        if slot is None:
            return None
        offset, size = slot
        try:
            with open(self.path, 'rb') as fh:
                fh.seek(offset)
                return fh.read(size)
        except Exception as exc:
            log.warning(f"ERF read {self.path} {name}: {exc}")
            return None

    def list_type(self, res_type: int) -> List[str]:
        suffix = f':{res_type}'
        return [k[:-(len(suffix))] for k in self._index if k.endswith(suffix)]

    def has(self, name: str, res_type: int) -> bool:
        return _key(name, res_type) in self._index


# ── ResourceManager ──────────────────────────────────────────────────────────

class ResourceManager:
    """
    Unified KotOR installation resource manager.

    Handles K1, K2, or both simultaneously.  Each installation is indexed
    independently; resource lookup checks the correct installation based on
    game tag, with cross-game fallback.

    Lookup priority per game (matches KotOR engine):
      1. Override/ loose files  (pre-loaded into memory for instant access)
      2. module ERFs (*.mod / *.rim / *.erf in modules/)
      3. TexturePacks ERFs       (for TPC/TGA only, TPA > TPB > TPC > GUI)
      4. streamed audio paths    (VO and sounds, bytes loaded on demand)
      5. chitin.key / BIF        (base game data)

    Thread safety
    -------------
    Installation indexes are immutable after construction. Mutable manager
    state (installation references and module overlays) is published under an
    RLock and advances :attr:`revision`. Strict model loads capture MDL and MDX
    bytes under one read lock, then release it before parsing.
    """

    def __init__(self):
        # PyKotor 2.3.1 wheels shipped a per-field debug print in
        # GFFStruct.acquire.  Apply the exact, process-local compatibility
        # guard before any Map Studio stock-module GFF hydration can run.
        self._pykotor_gff_runtime_fix_status = ensure_pykotor_gff_acquire_quiet()
        self._k1: Optional[_GameInstall] = None
        self._k2: Optional[_GameInstall] = None
        self._lock = threading.RLock()
        self._revision = 0
        # In-memory overlay for resources bundled inside a specific imported
        # module (custom community .mod/.rim/.erf that ships its own room
        # models, WOKs, and textures).  Checked before any game installation so
        # editing an imported custom map resolves its bundled assets. Keyed by
        # (resref_lower, res_type).
        self._overlay: Dict[Tuple[str, int], bytes] = {}
        # Replaceable in-memory resources owned by the active authored project.
        # These sit above imported-module/base-game content so custom textures
        # and models render in Map Studio before the module is exported.
        self._project_overlay: Dict[Tuple[str, int], bytes] = {}
        # Legacy community maps commonly ship a tiny ARE/GIT/IFO ``.mod`` in
        # ``Modules`` and put LYT/VIS/MDL/MDX/WOK/textures in a sibling
        # ``Override`` folder.  Keep those files lazy: several recovered map
        # sets are hundreds of megabytes and must not be decoded or retained
        # just to make the module discoverable in Map Studio.
        self._loose_overlay: Dict[Tuple[str, int], Path] = {}
        self._loose_overlay_candidates: Dict[Tuple[str, int], Tuple[Path, ...]] = {}

    @property
    def revision(self) -> int:
        """Monotonic resource-publication revision for dependent caches."""

        with self._lock:
            return int(self._revision)

    # ── Setup ────────────────────────────────────────────────────────────

    def set_k1_dir(self, path: str) -> bool:
        """Index a KotOR 1 installation directory. Returns True on success."""
        if not path or not os.path.isdir(path):
            return False
        try:
            inst = _GameInstall(path, 'K1')
            with self._lock:
                self._k1 = inst
                self._revision += 1
            log.info(f"ResourceManager: K1 indexed {path!r} — "
                     f"{len(inst._key_map)} key entries, "
                     f"{len(inst._tex_erfs)} tex ERFs, "
                     f"{len(inst._mod_erfs)} module ERFs, "
                     "streamed audio index deferred, "
                     f"{len(inst._override)} override files")
            return True
        except Exception as exc:
            log.error(f"ResourceManager: K1 index failed {path!r}: {exc}", exc_info=True)
            return False

    def set_k2_dir(self, path: str) -> bool:
        """Index a KotOR 2 installation directory. Returns True on success."""
        if not path or not os.path.isdir(path):
            return False
        try:
            inst = _GameInstall(path, 'K2')
            with self._lock:
                self._k2 = inst
                self._revision += 1
            log.info(f"ResourceManager: K2 indexed {path!r} — "
                     f"{len(inst._key_map)} key entries, "
                     f"{len(inst._tex_erfs)} tex ERFs, "
                     f"{len(inst._mod_erfs)} module ERFs, "
                     "streamed audio index deferred, "
                     f"{len(inst._override)} override files")
            return True
        except Exception as exc:
            log.error(f"ResourceManager: K2 index failed {path!r}: {exc}", exc_info=True)
            return False

    def add_module_overlay(self, capsule_path: str) -> int:
        """Index a custom module capsule (.mod/.rim/.erf) into the in-memory
        overlay so its bundled resources (room MDL/MDX/WOK, textures) resolve
        ahead of the base game.  Returns the number of resources added.

        Custom community modules ship their own room models rather than
        referencing base-game rooms; without this, converting their rooms to
        editable geometry fails ("could not be loaded from the game resources").
        """
        if not capsule_path or not os.path.isfile(capsule_path):
            return 0
        try:
            from pykotor.extract.capsule import LazyCapsule
        except Exception as exc:  # pragma: no cover - pykotor always present in app
            log.warning("ResourceManager.add_module_overlay: pykotor unavailable: %s", exc)
            return 0
        added = 0
        updates: Dict[Tuple[str, int], bytes] = {}
        try:
            for item in LazyCapsule(capsule_path):
                ext = str(item.restype().extension).lower()
                res_type = EXT_TO_TYPE.get(ext)
                if res_type is None:
                    continue
                data = bytes(item.data() or b"")
                if not data:
                    continue
                updates[(item.resref().lower(), res_type)] = data
                added += 1
        except Exception as exc:
            log.warning("ResourceManager.add_module_overlay: failed to index %r: %s", capsule_path, exc)
            return 0
        if updates:
            with self._lock:
                published = dict(self._overlay)
                published.update(updates)
                self._overlay = published
                self._revision += 1
        log.info("ResourceManager: module overlay indexed %d resources from %r", added, capsule_path)
        return added

    def set_project_overlay(self, resources: Any = ()) -> int:
        """Publish the active project's typed resources as a replaceable overlay.

        ``resources`` contains ``(resref, restype, bytes)`` rows. Replacing the
        complete map prevents a texture removed from one KMAP from leaking into
        the next project while preserving independently loaded module overlays.
        The manager revision advances only when the published bytes change.
        """

        published: Dict[Tuple[str, int], bytes] = {}
        for item in tuple(resources or ()):
            try:
                resref, raw_type, data = item
            except (TypeError, ValueError):
                continue
            name = str(resref or "").strip().lower()
            if isinstance(raw_type, int) and not isinstance(raw_type, bool):
                res_type = int(raw_type)
            else:
                res_type = EXT_TO_TYPE.get(
                    str(raw_type or "").strip().lower().lstrip("."),
                    -1,
                )
            payload = bytes(data or b"")
            if name and res_type >= 0 and payload:
                published[(name, res_type)] = payload
        with self._lock:
            if published != self._project_overlay:
                self._project_overlay = published
                self._revision += 1
        return len(published)

    def add_loose_overlay(self, directory: str, *, recursive: bool = True) -> int:
        """Index loose KOTOR resources from a recovered module bundle.

        Only extensions understood by :data:`EXT_TO_TYPE` are indexed and the
        bytes stay on disk until requested.  Later files win deterministically,
        with files inside an ``Override`` directory taking engine-like
        precedence over ancillary source folders in the same bundle.
        """

        root = Path(str(directory or "")).expanduser()
        if not root.is_dir():
            return 0
        try:
            candidates = root.rglob("*") if recursive else root.iterdir()
            files = [path for path in candidates if path.is_file()]
        except OSError as exc:
            log.warning("ResourceManager.add_loose_overlay: cannot scan %r: %s", str(root), exc)
            return 0

        def _priority(path: Path) -> tuple[int, int, int, str]:
            in_override = any(part.lower() == "override" for part in path.parts)
            in_binary_dir = any(part.lower() in {"bin", "bins", "binary", "compiled"} for part in path.parts)
            binary_mdl = False
            if path.suffix.lower() == ".mdl":
                try:
                    with path.open("rb") as stream:
                        prefix = stream.read(8)
                    binary_mdl = len(prefix) >= 4 and prefix[:4] == b"\x00\x00\x00\x00"
                except OSError:
                    binary_mdl = False
            return (
                1 if in_override else 0,
                1 if in_binary_dir else 0,
                1 if binary_mdl else 0,
                str(path).lower(),
            )

        added = 0
        candidates_by_key: Dict[Tuple[str, int], List[Path]] = {}
        selected_by_key: Dict[Tuple[str, int], Path] = {}
        for path in sorted(files, key=_priority):
            extension = path.suffix.lower().lstrip(".")
            res_type = EXT_TO_TYPE.get(extension)
            if res_type is None:
                continue
            resref = path.stem.strip().lower()
            if not resref:
                continue
            key = (resref, res_type)
            candidates_by_key.setdefault(key, []).append(path)
            selected_by_key[key] = path
            added += 1
        if selected_by_key:
            with self._lock:
                published_overlay = dict(self._loose_overlay)
                published_overlay.update(selected_by_key)
                published_candidates = dict(self._loose_overlay_candidates)
                for key, candidates in candidates_by_key.items():
                    prior = list(published_candidates.get(key, ()))
                    for candidate in candidates:
                        if candidate not in prior:
                            prior.append(candidate)
                    published_candidates[key] = tuple(prior)
                self._loose_overlay = published_overlay
                self._loose_overlay_candidates = published_candidates
                self._revision += 1
        log.info("ResourceManager: loose overlay indexed %d resources from %r", added, str(root))
        return added

    def _get_loose_overlay(self, name: str, res_type: int) -> Optional[bytes]:
        with self._lock:
            path = self._loose_overlay.get((str(name or "").lower(), int(res_type)))
        if path is None:
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            log.warning("ResourceManager: loose overlay read failed for %r: %s", str(path), exc)
            return None

    def overlay_source_path(self, name: str, res_type: int) -> str:
        """Return the physical loose-overlay source used for diagnostics."""

        with self._lock:
            path = self._loose_overlay.get((str(name or "").lower(), int(res_type)))
        return str(path) if path is not None else ""

    def overlay_candidate_paths(self, name: str, res_type: int) -> Tuple[str, ...]:
        """Return every same-resref loose candidate considered for diagnostics."""

        with self._lock:
            paths = self._loose_overlay_candidates.get((str(name or "").lower(), int(res_type)), ())
        return tuple(str(path) for path in paths)

    def clear_module_overlay(self) -> None:
        """Drop all bundled-module overlay resources."""
        with self._lock:
            self._overlay = {}
            self._project_overlay = {}
            self._loose_overlay = {}
            self._loose_overlay_candidates = {}
            self._revision += 1

    def get_k1(self) -> Optional['_GameInstall']:
        with self._lock:
            return self._k1

    def get_k2(self) -> Optional['_GameInstall']:
        with self._lock:
            return self._k2

    def is_ready(self) -> bool:
        """True if at least one installation is indexed."""
        with self._lock:
            return self._k1 is not None or self._k2 is not None

    # ── Resource access ──────────────────────────────────────────────────

    def get(self, name: str, res_type: int, game: str = 'K1') -> Optional[bytes]:
        """
        Fetch raw resource bytes.

        game: 'K1', 'K2', or 'auto' (tries game-tagged install first,
              then the other one as fallback).
        """
        # Bundled-module overlay wins (matches the engine's Override priority):
        # a custom module's own room models/WOKs/textures shadow the base game.
        authored = self._project_overlay.get((name.lower(), res_type))
        if authored is not None:
            return authored
        loose = self._get_loose_overlay(name, res_type)
        if loose is not None:
            return loose
        overlaid = self._overlay.get((name.lower(), res_type))
        if overlaid is not None:
            return overlaid
        inst = self._k1 if game == 'K1' else self._k2
        if inst is not None:
            data = inst.get(name, res_type)
            if data is not None:
                return data
        # Fallback: try the other installation
        other = self._k2 if game == 'K1' else self._k1
        if other is not None:
            data = other.get(name, res_type)
            if data is not None:
                return data
        return None

    def _get_strict_locked(self, name: str, res_type: int, game: str = 'K1') -> Optional[bytes]:
        """Fetch one strict resource while the caller holds ``self._lock``."""

        authored = self._project_overlay.get((name.lower(), res_type))
        if authored is not None:
            return authored
        loose = self._get_loose_overlay(name, res_type)
        if loose is not None:
            return loose
        overlaid = self._overlay.get((name.lower(), res_type))
        if overlaid is not None:
            return overlaid
        tag = str(game or 'K1').strip().upper()
        inst = self._k2 if tag == 'K2' else self._k1
        return inst.get(name, res_type) if inst is not None else None

    def get_strict(self, name: str, res_type: int, game: str = 'K1') -> Optional[bytes]:
        """Fetch only from the requested game (plus the active module overlay).

        Map Studio uses this for engine-facing template/model validation. A K1
        fallback while authoring K2 can produce a convincing preview for a UTP
        that does not exist in the target game.
        """

        with self._lock:
            return self._get_strict_locked(name, res_type, game)

    def get_mdl(self, name: str, game: str = 'K1') -> Optional[bytes]:
        return self.get(name, RES_MDL, game)

    def get_mdx(self, name: str, game: str = 'K1') -> Optional[bytes]:
        return self.get(name, RES_MDX, game)

    def get_texture(self, name: str, game: str = 'K1') -> Optional[bytes]:
        """Load texture: TPC first, then TGA."""
        for candidate in _texture_name_candidates(name):
            data = self.get(candidate, RES_TPC, game)
            if data is not None:
                return data
            data = self.get(candidate, RES_TGA, game)
            if data is not None:
                return data
        return None

    def get_txi(self, name: str, game: str = 'K1') -> str:
        """Return TXI string for texture name (empty string if absent)."""
        for candidate in _texture_name_candidates(name):
            raw = self.get(candidate, RES_TXI, game)
            if raw:
                try:
                    return raw.decode('ascii', 'replace')
                except Exception:
                    return ''
        return ''

    # ── Model listing ────────────────────────────────────────────────────

    def list_models(self, game: str = 'all') -> List[Tuple[str, str]]:
        """
        List all model resrefs as (resref, game_tag) tuples.
        game: 'K1', 'K2', or 'all'.
        """
        out: List[Tuple[str, str]] = []
        if game in ('K1', 'all') and self._k1:
            for r in self._k1.list_resrefs(RES_MDL):
                out.append((r, 'K1'))
        if game in ('K2', 'all') and self._k2:
            for r in self._k2.list_resrefs(RES_MDL):
                out.append((r, 'K2'))
        return out

    def list_textures(self, game: str = 'all') -> List[Tuple[str, str]]:
        """List all texture resrefs as (resref, game_tag) tuples."""
        out: List[Tuple[str, str]] = []
        if game in ('K1', 'all') and self._k1:
            for r in self._k1.list_resrefs(RES_TPC):
                out.append((r, 'K1'))
        if game in ('K2', 'all') and self._k2:
            for r in self._k2.list_resrefs(RES_TPC):
                out.append((r, 'K2'))
        return out

    def has_textures(self, game: str = 'all') -> bool:
        """Quick check: does any installation have texture data?

        Checks both TexturePacks ERF files AND BIF key_map entries (TPC/TGA).
        Most KotOR installations store textures in BIF archives via chitin.key —
        having TexturePacks ERFs is the exception, not the rule.  Previously this
        only checked _tex_erfs, which returned False for many valid installations,
        causing show_texture to never auto-enable.
        """
        # Check TexturePacks ERFs first (fast, common on modded installs)
        if game in ('K1', 'all') and self._k1:
            if self._k1._tex_erfs:
                return True
            # Also check BIF key_map for TPC (3007) or TGA (3) entries
            for k in self._k1._key_map:
                if k.endswith(f':{RES_TPC}') or k.endswith(f':{RES_TGA}'):
                    return True
        if game in ('K2', 'all') and self._k2:
            if self._k2._tex_erfs:
                return True
            for k in self._k2._key_map:
                if k.endswith(f':{RES_TPC}') or k.endswith(f':{RES_TGA}'):
                    return True
        return False

    # ── High-level loaders ───────────────────────────────────────────────

    def load_model(self, name: str, game: str = 'K1',
                   prefer_base_archive: bool = False):
        """
        Load and parse a model by resref name.
        Returns a KotorModel on success, None on failure.

        prefer_base_archive: read the KEY/BIF archive pair first, ignoring
        Override/ERF shadows.  Use this when the TRUE vanilla model is
        required (rig donors, vanilla baselines) — a user's own exported
        model in Override/ silently replaces the original otherwise.
        """
        mdl = None
        mdx = None
        source_layer = None
        if prefer_base_archive:
            inst = self._k1 if game == 'K1' else self._k2
            if inst is not None:
                mdl = inst.get_bif(name, RES_MDL)
                if mdl is not None:
                    mdx = inst.get_bif(name, RES_MDX) or b''
                    source_layer = 'base_game_archive'
        if mdl is None:
            mdl = self.get_mdl(name, game)
            if mdl is None:
                log.warning(f"ResourceManager.load_model: '{name}' not found in {game}")
                return None
            mdx = self.get_mdx(name, game) or b''
            source_layer = 'game_library'
        try:
            from ..game.kotor_loader import load_model_from_bytes
            model = load_model_from_bytes(mdl, mdx)
            if model is not None:
                model._gr_source_mdl_bytes = mdl
                model._gr_source_mdx_bytes = mdx
                model._gr_source_resref = name
                model._gr_source_game = game
                model._gr_source_layer = source_layer
            return model
        except Exception as exc:
            log.error(f"ResourceManager.load_model: parse failed for '{name}': {exc}",
                      exc_info=True)
            return None

    def load_model_strict(self, name: str, game: str = 'K1',
                          prefer_base_archive: bool = False):
        """Load a model without silently borrowing it from the other game.

        Engine-facing authoring and simulation previews must reflect the
        selected target installation.  The regular ``load_model`` API keeps
        its historical cross-game fallback for general browsing workflows;
        this explicit variant is the contract for Map Studio and supermodels.
        The active imported-module overlay remains eligible because it is part
        of the target module being edited.
        """

        tag = str(game or 'K1').strip().upper()
        if tag not in {'K1', 'K2'}:
            tag = 'K1'
        # Capture the binary pair under one resource-state revision. Parsing is
        # deliberately outside the lock because it is the expensive CPU phase.
        with self._lock:
            mdl = None
            mdx = None
            source_layer = None
            if prefer_base_archive:
                inst = self._k1 if tag == 'K1' else self._k2
                if inst is not None:
                    mdl = inst.get_bif(name, RES_MDL)
                    if mdl is not None:
                        mdx = inst.get_bif(name, RES_MDX) or b''
                        source_layer = 'base_game_archive'
            if mdl is None:
                mdl = self._get_strict_locked(name, RES_MDL, tag)
                if mdl is None:
                    log.warning(
                        "ResourceManager.load_model_strict: %r not found in %s",
                        name,
                        tag,
                    )
                    return None
                mdx = self._get_strict_locked(name, RES_MDX, tag) or b''
                source_layer = 'target_game_library'
        try:
            from ..game.kotor_loader import load_model_from_bytes

            model = load_model_from_bytes(mdl, mdx)
            if model is not None:
                model._gr_source_mdl_bytes = mdl
                model._gr_source_mdx_bytes = mdx
                model._gr_source_resref = name
                model._gr_source_game = tag
                model._gr_source_layer = source_layer
            return model
        except Exception as exc:
            log.error(
                "ResourceManager.load_model_strict: parse failed for %r: %s",
                name,
                exc,
                exc_info=True,
            )
            return None

    def load_texture_image(self, name: str, game: str = 'K1',
                           max_size: int = 512) -> Optional[object]:
        """
        Load a texture as a PIL Image (RGBA).
        Returns None if not found or not decodable.
        Applies resize to max_size if needed.
        """
        raw = self.get_texture(name, game)
        if raw is None:
            return None
        try:
            img = _decode_texture(raw)
            if img is None:
                return None
            if max_size and (img.width > max_size or img.height > max_size):
                img.thumbnail((max_size, max_size))
            return img
        except Exception as exc:
            log.debug(f"ResourceManager.load_texture_image: '{name}' decode failed: {exc}")
            return None

    def game_dir(self, game: str) -> Optional[str]:
        """Return the game directory for 'K1' or 'K2', or None if not set."""
        inst = self._k1 if game == 'K1' else self._k2
        return inst.game_dir if inst else None

    def stats(self) -> Dict[str, object]:
        """Return a dict of statistics for diagnostics."""
        result = {}
        for tag, inst in [('K1', self._k1), ('K2', self._k2)]:
            if inst:
                result[tag] = {
                    'dir': inst.game_dir,
                    'key_entries': len(inst._key_map),
                    'tex_erfs': len(inst._tex_erfs),
                    'mod_erfs': len(inst._mod_erfs),
                    'stream_sounds': sum(
                        1 for _path, res_type in inst._loose_audio.values()
                        if res_type == RES_WAV
                    ),
                    'stream_audio': len(inst._loose_audio),
                    'stream_audio_indexed': inst._loose_audio_indexed,
                    'override': len(inst._override),
                }
            else:
                result[tag] = None
        return result


# ── Per-game installation ────────────────────────────────────────────────────

class _GameInstall:
    """
    Internal class: indexes one KotOR installation (K1 or K2).

    Priority chain for get():
      1. _override dict  (indexed loose Override/ file paths, lazy read)
      2. _mod_erfs list  (module ERFs, lazy seek)
      3. _tex_erfs list  (TexturePacks ERFs for TPC/TGA, lazy seek)
      4. _loose_audio    (StreamVoice/StreamWaves/StreamSounds paths, lazy read)
      5. _bif_index dict (BIF files via chitin.key, lazy seek)
    """

    def __init__(self, game_dir: str, tag: str):
        import time
        self.game_dir = os.path.normpath(game_dir)
        self.tag = tag

        self._key_map: Dict[str, Tuple[int, int]] = {}    # _key → (bif_idx, var_idx)
        self._bif_index: Dict[int, _BifIndex] = {}        # bif_file_idx → _BifIndex
        self._tex_erfs: List[_ErfIndex] = []              # TexturePacks ERFs, TPA first
        self._mod_erfs: List[_ErfIndex] = []              # modules/ ERFs
        self._override: Dict[str, str] = {}               # Override/ loose file paths
        # Unified loose-audio index.  Values retain the concrete path and the
        # real resource type; bytes are not read or decoded during indexing.
        self._loose_audio: Dict[str, Tuple[str, int]] = {}
        self._loose_audio_indexed = False
        self._loose_audio_index_lock = threading.Lock()

        t0 = time.perf_counter()
        self._index_chitin()
        self._index_texture_packs()
        self._index_modules()
        self._load_override()
        elapsed = (time.perf_counter() - t0) * 1000
        log.debug(f"_GameInstall {tag} indexed {game_dir!r} in {elapsed:.0f}ms")

    @property
    def _stream_sounds(self) -> Dict[str, str]:
        """Read-only compatibility view of the former WAV-path index."""

        self._index_loose_audio()
        return {
            key: path
            for key, (path, res_type) in self._loose_audio.items()
            if res_type == RES_WAV
        }

    @_stream_sounds.setter
    def _stream_sounds(self, paths: Dict[str, str]) -> None:
        """Accept legacy test/caller setup while retaining one stored index."""

        self._loose_audio = {
            key: (path, RES_WAV)
            for key, path in paths.items()
        }
        self._loose_audio_indexed = False
        if not hasattr(self, '_loose_audio_index_lock'):
            self._loose_audio_index_lock = threading.Lock()

    # ── Indexing ──────────────────────────────────────────────────────────

    def _index_chitin(self):
        """Parse chitin.key → build _key_map and lazy-init _bif_index."""
        key_path = self._find_file('chitin.key')
        if not key_path:
            log.warning(f"_GameInstall {self.tag}: chitin.key not found in {self.game_dir!r}")
            return
        try:
            with open(key_path, 'rb') as fh:
                raw = fh.read()
        except OSError as exc:
            log.warning(f"_GameInstall: chitin.key read error: {exc}")
            return

        bif_count = struct.unpack_from('<I', raw, 8)[0]
        key_count = struct.unpack_from('<I', raw, 12)[0]
        off_bifs  = struct.unpack_from('<I', raw, 16)[0]
        off_keys  = struct.unpack_from('<I', raw, 20)[0]

        # Build BIF filename list
        bif_names: List[str] = []
        for i in range(bif_count):
            base = off_bifs + i * 12
            name_off = struct.unpack_from('<I', raw, base + 4)[0]
            name_sz  = struct.unpack_from('<H', raw, base + 8)[0]
            raw_name = raw[name_off:name_off + name_sz].split(b'\x00', 1)[0]
            name_str = raw_name.decode('ascii', 'replace').replace('\\', os.sep)
            bif_names.append(name_str)

        # Parse key entries: 22 bytes each — resref[16] type[2] id[4]
        key_raw = raw[off_keys: off_keys + key_count * 22]
        for i in range(key_count):
            b       = i * 22
            resref  = key_raw[b:b+16].split(b'\x00', 1)[0].decode('ascii', 'replace').lower()
            rtype   = struct.unpack_from('<H', key_raw, b + 16)[0]
            res_id  = struct.unpack_from('<I', key_raw, b + 18)[0]
            bif_idx = (res_id >> 20) & 0xFFF
            var_idx = res_id & 0xFFFFF
            self._key_map[_key(resref, rtype)] = (bif_idx, var_idx)

        # Lazy-init BIF readers
        for idx, name in enumerate(bif_names):
            full = os.path.join(self.game_dir, name)
            if os.path.isfile(full):
                self._bif_index[idx] = _BifIndex(full)
            else:
                found = self._find_file_ci(name)
                if found:
                    self._bif_index[idx] = _BifIndex(found)
                else:
                    log.debug(f"_GameInstall {self.tag}: BIF not found: {name}")

    def _index_texture_packs(self):
        """Index TexturePacks/ ERFs in priority order: TPA > TPB > TPC > GUI."""
        tp_dir = self._find_dir('TexturePacks') or self._find_dir('texturepacks')
        if not tp_dir:
            return
        priority_order = ['tpa', 'tpb', 'tpc', 'gui']
        try:
            erf_files = sorted(
                [f for f in os.listdir(tp_dir) if f.lower().endswith('.erf')],
                key=lambda f: next(
                    (i for i, p in enumerate(priority_order) if p in f.lower()), 99
                )
            )
        except OSError:
            return
        for fname in erf_files:
            path = os.path.join(tp_dir, fname)
            idx = _ErfIndex(path)
            if idx._index:
                self._tex_erfs.append(idx)
                log.debug(f"_GameInstall {self.tag}: tex ERF {fname} "
                          f"({len(idx._index)} entries)")

    def _index_modules(self):
        """Index module ERFs (*.mod, *.rim, *.erf) in modules/."""
        mod_dir = self._find_dir('modules') or self._find_dir('Modules')
        if not mod_dir:
            return
        try:
            files = os.listdir(mod_dir)
        except OSError:
            return
        for fname in sorted(files):
            if fname.lower().endswith(('.mod', '.rim', '.erf')):
                path = os.path.join(mod_dir, fname)
                idx = _ErfIndex(path)
                if idx._index:
                    self._mod_erfs.append(idx)
        log.debug(f"_GameInstall {self.tag}: {len(self._mod_erfs)} module ERFs")

    def _load_override(self):
        """Index Override/ loose files without loading their contents."""
        ovr_dir = self._find_dir('Override') or self._find_dir('override')
        if not ovr_dir:
            return
        loaded = 0
        try:
            for fname in os.listdir(ovr_dir):
                path = os.path.join(ovr_dir, fname)
                if not os.path.isfile(path):
                    continue
                base, ext = os.path.splitext(fname.lower())
                rtype = EXT_TO_TYPE.get(ext.lstrip('.'))
                if rtype is None:
                    continue
                self._override[_key(base, rtype)] = path
                loaded += 1
        except OSError:
            pass
        if loaded:
            log.debug(f"_GameInstall {self.tag}: {loaded} Override files indexed")

    def _index_loose_audio(self) -> None:
        """Populate the path-only audio index once, on first audio access."""

        if getattr(self, '_loose_audio_indexed', False):
            return
        lock = getattr(self, '_loose_audio_index_lock', None)
        if lock is None:
            lock = threading.Lock()
            self._loose_audio_index_lock = lock
        with lock:
            if getattr(self, '_loose_audio_indexed', False):
                return
            self._scan_loose_audio()
            self._loose_audio_indexed = True

    def _scan_loose_audio(self) -> None:
        """Index streamed KOTOR audio paths without reading or decoding bytes.

        KOTOR 1 stores dialogue voice-over recursively under ``StreamWaves``;
        KOTOR 2 uses ``StreamVoice``.  A few installations contain both names,
        so the target game's canonical VO directory wins, followed by the
        alternate VO name and then ``StreamSounds``.  That final tier keeps
        ambient UTS audio available without allowing a same-named ambient clip
        to shadow dialogue VO.  Directory and extension matching is
        case-insensitive.

        ``.wav`` and genuine ``.mp3`` files retain their distinct resource
        types.  Only paths are indexed; file contents remain on disk until
        :meth:`get` resolves a requested resource.
        """

        tag = str(self.tag or '').strip().upper()
        directory_names = (
            ('StreamWaves', 'StreamVoice', 'StreamSounds')
            if tag == 'K1'
            else ('StreamVoice', 'StreamWaves', 'StreamSounds')
        )
        indexed = 0
        for directory_name in directory_names:
            stream_dir = self._find_dir(directory_name)
            if not stream_dir:
                continue
            try:
                for current_dir, child_dirs, files in os.walk(stream_dir):
                    # Deterministic traversal makes duplicate-resref precedence
                    # stable even on case-insensitive filesystems.
                    child_dirs.sort(key=str.lower)
                    files.sort(key=str.lower)
                    for filename in files:
                        base, ext = os.path.splitext(filename)
                        res_type = _LOOSE_AUDIO_EXT_TYPES.get(ext.lower())
                        if res_type is None:
                            continue
                        key = _key(base, res_type)
                        if key in self._loose_audio:
                            continue
                        path = os.path.join(current_dir, filename)
                        if not os.path.isfile(path):
                            continue
                        self._loose_audio[key] = (path, res_type)
                        indexed += 1
            except OSError:
                continue
        if indexed:
            log.debug(f"_GameInstall {self.tag}: {indexed} streamed audio files indexed")

    def _index_stream_sounds(self) -> None:
        """Compatibility wrapper for the former StreamSounds-only indexer."""

        if not hasattr(self, '_loose_audio'):
            self._loose_audio = {}
        self._index_loose_audio()

    # ── Resource access ───────────────────────────────────────────────────

    def get(self, name: str, res_type: int) -> Optional[bytes]:
        """
        Fetch raw resource bytes by name + type.
        Priority: Override > modules ERF > TexturePacks ERF > streamed audio > BIF.
        """
        k = _key(name, res_type)

        # 1. Override
        override_path = self._override.get(k)
        if override_path is not None:
            try:
                with open(override_path, 'rb') as fh:
                    return fh.read()
            except OSError:
                return None

        # 2. Module ERFs (area-specific resources)
        for erf in self._mod_erfs:
            if erf.has(name, res_type):
                data = erf.read(name, res_type)
                if data is not None:
                    return data

        # 3. TexturePacks ERFs (for TPC/TGA/TXI only — texture-specific)
        if res_type in (RES_TPC, RES_TGA, RES_TXI):
            for erf in self._tex_erfs:
                data = erf.read(name, res_type)
                if data is not None:
                    return data

        # 4. StreamVoice/StreamWaves/StreamSounds loose audio. Authored module
        # resources above retain precedence over stock installation audio.
        if res_type in (RES_WAV, RES_MP3):
            self._index_loose_audio()
            loose_audio = self._loose_audio.get(k)
            if loose_audio is not None:
                stream_path, indexed_type = loose_audio
                if indexed_type == res_type:
                    try:
                        with open(stream_path, 'rb') as fh:
                            return fh.read()
                    except OSError:
                        # A stale/deleted path may still have a valid BIF fallback.
                        pass

        # 5. BIF via chitin.key
        slot = self._key_map.get(k)
        if slot is not None:
            bif_idx, var_idx = slot
            bif = self._bif_index.get(bif_idx)
            if bif is not None:
                return bif.read(var_idx)

        return None

    def get_bif(self, name: str, res_type: int) -> Optional[bytes]:
        """Fetch KEY/BIF bytes only, ignoring Override/ERF shadows."""
        slot = self._key_map.get(_key(name, res_type))
        if slot is None:
            return None
        bif_idx, var_idx = slot
        bif = self._bif_index.get(bif_idx)
        if bif is None:
            return None
        return bif.read(var_idx)

    def has(self, name: str, res_type: int) -> bool:
        k = _key(name, res_type)
        if k in self._override:
            return True
        for erf in self._mod_erfs:
            if erf.has(name, res_type):
                return True
        if res_type in (RES_TPC, RES_TGA, RES_TXI):
            for erf in self._tex_erfs:
                if erf.has(name, res_type):
                    return True
        if res_type in (RES_WAV, RES_MP3):
            self._index_loose_audio()
            if k in self._loose_audio:
                return True
        return k in self._key_map

    def list_resrefs(self, res_type: int) -> List[str]:
        """List all known resource names of given type (sorted, deduped)."""
        out: Set[str] = set()
        suffix = f':{res_type}'
        for k in self._override:
            if k.endswith(suffix):
                out.add(k[:-(len(suffix))])
        for erf in self._mod_erfs:
            for r in erf.list_type(res_type):
                out.add(r)
        for erf in self._tex_erfs:
            for r in erf.list_type(res_type):
                out.add(r)
        if res_type in (RES_WAV, RES_MP3):
            self._index_loose_audio()
            for k in self._loose_audio:
                if k.endswith(suffix):
                    out.add(k[:-(len(suffix))])
        for k in self._key_map:
            if k.endswith(suffix):
                out.add(k[:-(len(suffix))])
        return sorted(out)

    # ── Path helpers ──────────────────────────────────────────────────────

    def _find_file(self, name: str) -> Optional[str]:
        """Find a file directly under game_dir, case-insensitively."""
        exact = os.path.join(self.game_dir, name)
        if os.path.isfile(exact):
            return exact
        try:
            entries = {e.lower(): e for e in os.listdir(self.game_dir)}
            real = entries.get(name.lower())
            if real:
                return os.path.join(self.game_dir, real)
        except OSError:
            pass
        return None

    def _find_dir(self, name: str) -> Optional[str]:
        """Find a subdirectory under game_dir, case-insensitively."""
        exact = os.path.join(self.game_dir, name)
        if os.path.isdir(exact):
            return exact
        try:
            entries = {e.lower(): e for e in os.listdir(self.game_dir)}
            real = entries.get(name.lower())
            if real:
                candidate = os.path.join(self.game_dir, real)
                if os.path.isdir(candidate):
                    return candidate
        except OSError:
            pass
        return None

    def _find_file_ci(self, rel_path: str) -> Optional[str]:
        """Find a file via case-insensitive multi-component path walk."""
        parts = rel_path.replace('\\', '/').split('/')
        current = self.game_dir
        for part in parts:
            if not os.path.isdir(current):
                return None
            try:
                entries = {e.lower(): e for e in os.listdir(current)}
                real = entries.get(part.lower())
                if real is None:
                    return None
                current = os.path.join(current, real)
            except OSError:
                return None
        return current if os.path.isfile(current) else None


# ── Texture decoding helper ──────────────────────────────────────────────────


def _decode_texture(raw: bytes) -> Optional[object]:
    """Decode raw texture bytes to a PIL RGBA Image.

    Primary decoder: **PyKotor** (pykotor.resource.formats.tpc).
    PyKotor's ``read_tpc()`` auto-detects the format (TPC / TGA / DDS)
    and handles every KotOR texture encoding:

      • TPC binary — DXT1, DXT3, DXT5, RGB, RGBA, Greyscale, BGRA
      • TGA        — uncompressed, RLE, colour-mapped, true-colour
      • DDS        — S3TC / DXT compression

    Decoding pipeline (all from pykotor source):
      ``read_tpc(raw)``  →  ``TPC.convert(RGBA)``  →  ``TPCMipmap.to_pil_image()``

    Cross-references:
      • pykotor.resource.formats.tpc.tpc_auto.read_tpc
      • pykotor.resource.formats.tpc.tpc_data.TPC.convert
      • pykotor.resource.formats.tpc.tpc_data.TPCMipmap.to_pil_image
      • pykotor.gl.shader.texture.Texture.from_tpc  (PyKotor GL upload)

    The returned image is always **bottom-up** (row 0 = bottom of texture,
    OpenGL convention).  DXT-compressed textures are flipped from PyKotor's
    top-down output to bottom-up.  This matches viewport.py's _load_tpc_bytes
    contract so that gpu_renderer._upload() can upload without any flip, and
    the vertex shader's ``v_uv.y = 1.0 - in_uv.y`` produces correct results.
    Phase D11 fix: previously returned top-down, causing upside-down textures
    when loaded through the ResourceManager → GPU renderer path.

    Only falls back to PIL.Image.open for non-KotOR formats (PNG, BMP)
    that PyKotor does not handle.

    Returns None if decoding fails entirely.
    """
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        return None

    if not raw:
        return None

    # ── Primary: PyKotor read_tpc (handles TPC, TGA, DDS) ───────────────
    # Apply header patch first (stock KotOR DXT files have data_size=0).
    try:
        from ..game.kotor_loader import patch_tpc_header
        patched = patch_tpc_header(raw)
    except ImportError:
        patched = raw

    try:
        from pykotor.resource.formats.tpc.tpc_auto import read_tpc as _pk_read
        from pykotor.resource.formats.tpc.tpc_data import TPCTextureFormat as _Fmt

        tpc = _pk_read(patched)
        # FIX-VFLIP-D11: Detect DXT-compressed format BEFORE conversion.
        # PyKotor's to_pil_image() returns:
        #   DXT1/DXT3/DXT5:  TOP-DOWN (DirectX DXT block order)
        #   Uncompressed:    BOTTOM-UP (OpenGL convention)
        # The GPU renderer's _upload() expects ALL images in BOTTOM-UP
        # (OpenGL) orientation and does NO flip on upload.  The vertex
        # shader applies v_uv.y = 1.0 - in_uv.y to convert KotOR's
        # V=0=top convention to GL V=0=bottom.  This only works correctly
        # when the image data is bottom-up.
        #
        # viewport.py _load_tpc_bytes already applies this flip for its
        # TextureCache path.  _decode_texture must match to avoid
        # top-down textures being sampled upside-down via the
        # ResourceManager → GPU renderer path.
        #
        # Reference: viewport.py _load_tpc_bytes line 558; gpu_renderer.py
        # _GlTexCache._upload docstring; KotOR.js TextureLoader.ts.
        _orig_fmt = tpc.format()
        _is_dxt = _orig_fmt in (
            _Fmt.DXT1, _Fmt.DXT3, _Fmt.DXT5
        ) if all(hasattr(_Fmt, x) for x in ('DXT1', 'DXT3', 'DXT5')) else False
        # Fallback detection: check raw TPC encoding byte at offset 12
        if not _is_dxt and len(raw) > 12:
            _enc_byte = raw[12]
            _data_sz_val = struct.unpack_from('<I', raw, 0)[0] if len(raw) >= 4 else 0
            # enc=2 with data_sz!=0 or pixel data smaller than uncompressed → DXT1
            # enc=4 with data_sz!=0 or pixel data smaller than uncompressed → DXT5
            if _enc_byte in (2, 4, 10, 12, 13, 14):
                _w = struct.unpack_from('<H', raw, 8)[0] if len(raw) >= 10 else 0
                _h = struct.unpack_from('<H', raw, 10)[0] if len(raw) >= 12 else 0
                _pixel_data_len = len(raw) - 128
                _uncompressed_min = {1: _w*_h, 2: _w*_h*3, 4: _w*_h*4}.get(_enc_byte, _w*_h*4)
                if _data_sz_val != 0 or (_w > 0 and _h > 0 and _pixel_data_len < _uncompressed_min):
                    _is_dxt = True

        tpc.convert(_Fmt.RGBA)
        img = tpc.get(0, 0).to_pil_image()
        if img is not None:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            # FIX-VFLIP-D11: Flip DXT textures from top-down to bottom-up
            # so the GPU renderer's upload + vertex shader V-flip chain
            # produces correct orientation.  Uncompressed textures are
            # already bottom-up from PyKotor — no flip needed.
            if _is_dxt:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Attach TXI metadata from the TPC object
            txi = ''
            try:
                txi = (tpc.txi or '').strip() if isinstance(getattr(tpc, 'txi', None), str) else ''
            except Exception:
                pass
            manual_txi = _tpc_uncompressed_txi(raw)
            if manual_txi:
                txi = manual_txi
            img._txi_str = txi                        # type: ignore[attr-defined]
            img._tpc_raw = raw                         # type: ignore[attr-defined]
            try:
                at = struct.unpack_from('<f', raw, 4)[0]
                img._txi_alpha_test = at if 0.0 < at <= 1.0 else None  # type: ignore[attr-defined]
            except Exception:
                img._txi_alpha_test = None             # type: ignore[attr-defined]
            return img
    except Exception as _pk_err:
        log.debug("_decode_texture: PyKotor read_tpc failed (%s)", _pk_err)

    # ── PIL direct: PNG, BMP, etc. (non-KotOR formats only) ──────────────
    try:
        img = Image.open(_io.BytesIO(raw)).convert('RGBA')
        return img
    except Exception:
        return None


def _tpc_uncompressed_txi(raw: bytes) -> str:
    """Return TXI text for uncompressed TPC payloads, or empty string."""
    if not _is_tpc(raw):
        return ''
    try:
        data_size = struct.unpack_from('<I', raw, 0)[0]
        if data_size != 0:
            return ''
        width = struct.unpack_from('<H', raw, 8)[0]
        height = struct.unpack_from('<H', raw, 10)[0]
        encoding = raw[12]
        mip_count = max(1, raw[13])
        bpp = {1: 1, 2: 3, 4: 4, 12: 4}.get(encoding)
        if bpp is None or width <= 0 or height <= 0:
            return ''
        total = 0
        w, h = width, height
        for _ in range(mip_count):
            total += max(1, w) * max(1, h) * bpp
            w = max(1, w >> 1)
            h = max(1, h >> 1)
        start = 128 + total
        if start >= len(raw):
            return ''
        txi = raw[start:].rstrip(b'\x00').decode('utf-8', errors='replace').strip()
        if not txi:
            return ''
        first = txi.splitlines()[0].split()
        if first and first[0].isascii() and first[0].isalpha():
            return txi
    except Exception:
        return ''
    return ''


def tpc_info(raw: bytes) -> Optional[Dict[str, Any]]:
    """Return a dict of TPC header fields, or None if *raw* is not a TPC file.

    Useful for diagnostics and migration testing.  Fields returned:
      ``data_size``, ``alpha_test``, ``width``, ``height``,
      ``encoding``, ``num_mips``, ``is_compressed``,
      ``format`` (human-readable string: 'DXT1', 'DXT5', 'Grey', 'RGB', 'RGBA').
    """
    if not _is_tpc(raw):
        return None
    data_size = struct.unpack_from('<I', raw, 0)[0]
    alpha_test = struct.unpack_from('<f', raw, 4)[0]
    width  = struct.unpack_from('<H', raw, 8)[0]
    height = struct.unpack_from('<H', raw, 10)[0]
    encoding = raw[12]
    num_mips = raw[13]
    compressed = data_size > 0
    fmt_map = {1: 'Grey', 2: ('DXT1' if compressed else 'RGB'),
               4: ('DXT5' if compressed else 'RGBA'), 12: 'RGBA'}
    return {
        'data_size':   data_size,
        'alpha_test':  alpha_test,
        'width':       width,
        'height':      height,
        'encoding':    encoding,
        'num_mips':    num_mips,
        'is_compressed': compressed,
        'format':      fmt_map.get(encoding, f'unknown({encoding})'),
    }


def _is_tpc(raw: bytes) -> bool:
    """Heuristic: is this a KotOR TPC file?

    Checks the 128-byte TPC header for valid encoding ID, non-zero power-of-two
    dimensions, and consistent data_size / encoding pairing.  Also rejects data
    that starts with PNG/JPEG/BMP/DDS magic bytes since those cannot be TPC.
    """
    if len(raw) < 128:
        return False

    # Fast-reject: known non-TPC magic bytes
    # PNG: 0x89 50 4E 47 | JPEG: FF D8 FF | BMP: 42 4D | DDS: 44 44 53 20
    if raw[:4] in (b'\x89PNG', b'\xff\xd8\xff\xe0', b'\xff\xd8\xff\xe1',
                   b'DDS ', b'BM\x00\x00'):
        return False

    # TPC header layout (all little-endian):
    #   offset 0  : uint32 data_size   — 0 for uncompressed, >0 for DXT
    #   offset 4  : float  alpha_test
    #   offset 8  : uint16 width
    #   offset 10 : uint16 height
    #   offset 12 : uint8  encoding   (1=Grey, 2=RGB/DXT1, 4=RGBA/DXT5, 12=RGBA)
    #   offset 13 : uint8  num_mips
    data_size = struct.unpack_from('<I', raw, 0)[0]
    width     = struct.unpack_from('<H', raw, 8)[0]
    height    = struct.unpack_from('<H', raw, 10)[0]
    encoding  = raw[12]
    num_mips  = raw[13]

    # Valid encoding IDs
    if encoding not in (1, 2, 4, 12):
        return False

    # Dimensions must be non-zero and power-of-two ≤ 4096
    if width == 0 or height == 0 or width > 4096 or height > 4096:
        return False
    if (width & (width - 1)) != 0 or (height & (height - 1)) != 0:
        return False

    # For compressed formats (data_size > 0) only encoding 2 (DXT1) or 4 (DXT5)
    # are valid.  Encoding 1 (Grey) or 12 (RGBA) are always uncompressed.
    if data_size > 0 and encoding not in (2, 4):
        return False

    # Sanity: num_mips should be between 1 and 13 (log2(4096)+1)
    if num_mips == 0 or num_mips > 13:
        return False

    # Total file size lower-bound check
    if data_size > 0:
        if len(raw) < 128 + data_size:
            return False
    else:
        min_pixels = width * height
        bytes_per_px = {1: 1, 2: 3, 4: 4, 12: 4}.get(encoding, 4)
        if len(raw) < 128 + min_pixels * bytes_per_px:
            return False

    return True





# ── Module-level singleton ───────────────────────────────────────────────────

# Global singleton — created once, shared by all components
_manager: Optional[ResourceManager] = None
_manager_lock = threading.Lock()


def get_manager() -> ResourceManager:
    """Get (or create) the global ResourceManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ResourceManager()
    return _manager


def reset_manager() -> ResourceManager:
    """Reset the global singleton (useful for testing)."""
    global _manager
    with _manager_lock:
        _manager = ResourceManager()
    return _manager


# ── Headless texture resolver ───────────────────────────────────────────────

def resolve_model_textures(model, manager: Optional[ResourceManager] = None,
                            game: str = 'K1',
                            max_size: int = 512) -> Dict[str, Any]:
    """
    Load all textures referenced by *model* from the ResourceManager.

    Returns a dict mapping **lowercased** texture name → PIL RGBA Image,
    suitable for passing directly to ``GpuRenderer.render(textures=...)``.

    This is the **headless** equivalent of what the live viewport does via
    its ``TextureCache`` + prewarm thread: it walks every mesh node,
    collects diffuse / lightmap / env-map / specular / bump texture names,
    loads each TPC/TGA from the ResourceManager priority chain
    (Override → module ERFs → TexturePacks → BIF), decodes to PIL RGBA,
    and applies KotOR-specific alpha processing (bump-opaque, punchthrough,
    env-blend-over).

    Pipeline (mirrors KotOR engine / xoreos / KotOR.js):
      node.texture      → diffuse texture  (textureMap1 in KotOR.js)
      node.lightmap      → lightmap texture (textureMap2 in KotOR.js)
      node.txi_envmaptexture → environment map
      node.txi_specularcolour → specular colour map
      node.txi_bumpmaptexture → bump/normal map
      node.texture_names → additional per-material textures

    Args:
        model:    KotorModel (from src.core.model_data)
        manager:  ResourceManager instance; if None, uses the global singleton.
        game:     'K1' or 'K2'
        max_size: max texture dimension (default 512)

    Returns:
        dict[str, PIL.Image.Image]  (lowercased name → RGBA image)
    """
    if manager is None:
        manager = get_manager()
    if not manager.is_ready():
        log.warning("resolve_model_textures: ResourceManager not ready (no game dir set)")
        return {}
    if model is None:
        return {}

    textures: Dict[str, Any] = {}
    tex_names: set = set()

    # Collect all texture names from model nodes
    all_nodes_fn = getattr(model, 'all_nodes', None)
    nodes = list(all_nodes_fn()) if all_nodes_fn else []

    for node in nodes:
        if not getattr(node, 'is_mesh', False):
            continue
        # Primary diffuse texture
        tex = str(getattr(node, 'texture', '') or '').strip()
        if tex and tex.upper() not in ('NULL', '', 'NONE'):
            tex_names.add(tex.lower())
        # Lightmap texture
        lm = str(getattr(node, 'lightmap', '') or '').strip()
        if lm and lm.upper() not in ('NULL', '', 'NONE'):
            tex_names.add(lm.lower())
        # Environment map
        env = str(getattr(node, 'txi_envmaptexture', '') or '').strip()
        if env and env.upper() not in ('NULL', '', 'NONE'):
            tex_names.add(env.lower())
        # Specular colour map
        spec = str(getattr(node, 'txi_specularcolour', '') or '').strip()
        if spec and spec.upper() not in ('NULL', '', 'NONE'):
            tex_names.add(spec.lower())
        # Bump map
        bump = str(getattr(node, 'txi_bumpmaptexture', '') or '').strip()
        if bump and bump.upper() not in ('NULL', '', 'NONE'):
            tex_names.add(bump.lower())
        # Additional per-material texture names
        for tn in getattr(node, 'texture_names', []):
            tn_clean = str(tn or '').strip()
            if tn_clean and tn_clean.upper() not in ('NULL', '', 'NONE'):
                tex_names.add(tn_clean.lower())

    # Load each texture
    loaded = 0
    missing = []
    for name in sorted(tex_names):
        raw = manager.get_texture(name, game)
        if raw is None:
            missing.append(name)
            continue
        try:
            img = _decode_texture(raw)
            if img is None:
                missing.append(name)
                continue
            if max_size and (img.width > max_size or img.height > max_size):
                img.thumbnail((max_size, max_size))
            # Apply KotOR alpha processing
            # Get TXI for alpha mode detection
            txi_str = getattr(img, '_txi_str', None) or ''
            if not txi_str:
                txi_str = manager.get_txi(name, game)
            txi_meta = _parse_txi_for_alpha(txi_str)
            img = _apply_alpha_fix(raw, img, txi_meta)
            textures[name] = img
            loaded += 1
        except Exception as exc:
            log.debug(f"resolve_model_textures: '{name}' decode failed: {exc}")
            missing.append(name)

    if missing:
        log.info(f"resolve_model_textures: {loaded} loaded, "
                 f"{len(missing)} missing: {missing[:10]}")
    else:
        log.info(f"resolve_model_textures: {loaded} textures loaded successfully")

    return textures


def audit_model_textures(model, manager: Optional[ResourceManager] = None,
                         game: str = 'K1') -> Dict[str, Any]:
    """
    Audit all textures referenced by a model — returns structured report.

    Provides clear error reporting for every texture: found/missing,
    source archive, size, format, and per-node mapping.

    Returns:
        dict with keys:
          - model_name: str
          - node_count / mesh_count: int
          - textures_expected: list of names
          - textures_found: dict[name → {size, format, source}]
          - textures_missing: list of names
          - per_node: list of {node_name, diffuse, lightmap, envmap, ...}
    """
    if manager is None:
        manager = get_manager()
    if model is None:
        return {"model_name": None, "error": "No model provided"}
    if not manager.is_ready():
        return {"model_name": getattr(model, 'name', '?'),
                "error": "ResourceManager not ready (no game dir set)"}

    model_name = getattr(model, 'name', getattr(model, 'model_name', '?'))
    all_nodes_fn = getattr(model, 'all_nodes', None)
    nodes = list(all_nodes_fn()) if all_nodes_fn else []

    tex_names: set = set()
    per_node = []

    for node in nodes:
        if not getattr(node, 'is_mesh', False):
            continue
        entry = {"node_name": getattr(node, 'name', '?')}
        for attr, key in [('texture', 'diffuse'), ('lightmap', 'lightmap'),
                          ('txi_envmaptexture', 'envmap'),
                          ('txi_specularcolour', 'specular'),
                          ('txi_bumpmaptexture', 'bumpmap')]:
            val = str(getattr(node, attr, '') or '').strip()
            if val and val.upper() not in ('NULL', '', 'NONE'):
                entry[key] = val.lower()
                tex_names.add(val.lower())
            else:
                entry[key] = None
        per_node.append(entry)

    textures_found = {}
    textures_missing = []

    for name in sorted(tex_names):
        raw_tpc = manager.get(name, RES_TPC, game)
        raw_tga = manager.get(name, RES_TGA, game)
        raw = raw_tpc or raw_tga
        if raw is not None:
            fmt = "TPC" if raw_tpc else "TGA"
            # Determine source: check priority chain
            source = _identify_texture_source(name, manager, game)
            textures_found[name] = {
                "size": len(raw),
                "format": fmt,
                "source": source,
            }
            # Try to decode for dimensions
            try:
                img = _decode_texture(raw)
                if img:
                    textures_found[name]["width"] = img.width
                    textures_found[name]["height"] = img.height
            except Exception:
                pass
        else:
            textures_missing.append(name)
            log.warning(f"TEXTURE MISSING: '{name}' — not found in Override, "
                        f"module ERFs, TexturePacks, or BIF archives "
                        f"(game={game})")

    return {
        "model_name": model_name,
        "node_count": len(nodes),
        "mesh_count": len(per_node),
        "textures_expected": sorted(tex_names),
        "textures_found": textures_found,
        "textures_missing": textures_missing,
        "textures_found_count": len(textures_found),
        "textures_missing_count": len(textures_missing),
        "per_node": per_node,
    }


def _identify_texture_source(name: str, manager: ResourceManager,
                              game: str = 'K1') -> str:
    """Identify which archive a texture was loaded from (for error reporting)."""
    inst = manager._k1 if game == 'K1' else manager._k2
    if inst is None:
        return "unknown"

    k_tpc = _key(name, RES_TPC)
    k_tga = _key(name, RES_TGA)

    # 1. Override
    if k_tpc in inst._override or k_tga in inst._override:
        return "Override/"

    # 2. Module ERFs
    for erf in inst._mod_erfs:
        if erf.has(name, RES_TPC) or erf.has(name, RES_TGA):
            return f"module ERF ({os.path.basename(erf.path)})"

    # 3. TexturePacks ERFs
    for erf in inst._tex_erfs:
        if erf.has(name, RES_TPC) or erf.has(name, RES_TGA):
            return f"TexturePack ({os.path.basename(erf.path)})"

    # 4. BIF
    for k in (k_tpc, k_tga):
        slot = inst._key_map.get(k)
        if slot is not None:
            bif_idx, _ = slot
            bif = inst._bif_index.get(bif_idx)
            if bif is not None:
                return f"BIF ({os.path.basename(bif.path)})"
            else:
                return f"BIF index {bif_idx} (file not available)"

    return "unknown"


def _parse_txi_for_alpha(txi_str: str) -> dict:
    """Minimal TXI parser for alpha-processing fields."""
    meta: Dict[str, Any] = {'blending': 0}
    if not txi_str:
        return meta
    for line in txi_str.splitlines():
        line = line.strip().lower()
        if line.startswith('blending '):
            parts = line.split()
            if len(parts) >= 2:
                val = parts[1]
                if val == 'punchthrough':
                    meta['blending'] = 2
                elif val == 'additive':
                    meta['blending'] = 1
                else:
                    try:
                        meta['blending'] = int(val)
                    except ValueError:
                        pass
        elif line.startswith('bumpmaptexture '):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                meta['bumpmaptexture'] = parts[1]
        elif line.startswith('envmaptexture '):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                meta['envmaptexture'] = parts[1]
        elif line.startswith('bumpyshinytexture '):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                meta['envmaptexture'] = parts[1]
    return meta


def _apply_alpha_fix(raw_bytes: bytes, img, txi_meta: dict):
    """Apply KotOR alpha processing to loaded texture (mirrors viewport logic).

    Rules (matches gpu_renderer.py / viewport.py _apply_kotor_alpha):
      1. bumpmaptexture → force alpha=255 (bump data in alpha channel)
      2. envmaptexture  → preserve alpha (env blend weight)
      3. blending=punchthrough → binary alpha cutoff
      4. blending=additive     → preserve alpha
      5. Standard (no bump, no env, blending=0) → force alpha=255
    """
    try:
        import numpy as np
    except ImportError:
        return img
    if img is None:
        return img

    blending = txi_meta.get('blending', 0)
    has_bump = bool(txi_meta.get('bumpmaptexture', ''))
    has_env = bool(txi_meta.get('envmaptexture', ''))

    if has_bump:
        # Bump map alpha = normal data, not transparency
        arr = np.array(img)
        arr[:, :, 3] = 255
        from PIL import Image
        return Image.fromarray(arr, 'RGBA')
    elif has_env:
        # Preserve alpha for env-map blend weight
        return img
    elif blending == 2:
        # Punchthrough: binary alpha
        try:
            import struct as _st
            alpha_thresh = _st.unpack_from('<f', raw_bytes, 4)[0]
            if not (0.0 < alpha_thresh <= 1.0):
                alpha_thresh = 0.5
        except Exception:
            alpha_thresh = 0.5
        arr = np.array(img)
        mask = arr[:, :, 3] < int(alpha_thresh * 255)
        arr[mask, 3] = 0
        arr[~mask, 3] = 255
        from PIL import Image
        return Image.fromarray(arr, 'RGBA')
    elif blending == 1:
        # Additive: keep alpha as-is
        return img
    else:
        # Standard: force opaque (KotOR stores specular/bump in alpha)
        arr = np.array(img)
        arr[:, :, 3] = 255
        from PIL import Image
        return Image.fromarray(arr, 'RGBA')

    return img
