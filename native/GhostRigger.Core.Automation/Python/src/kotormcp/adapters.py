"""
KotorMCP Adapters — concrete implementations of the port contracts.

Design rationale (Khononov, "Balancing Coupling in Software Design"):

  • Intrusive/Functional coupling is ALLOWED here — these adapters are the
    designated "knowledge sink" for pykotor internals. All high-volatility
    pykotor API calls live in this one module.

  • Tool handlers depend only on the Port contracts (low-volatility), not
    on these adapters. This satisfies the Distance × Strength × Volatility
    balance: strong coupling is kept at short distance (same sub-package)
    to a well-understood stable adapter, not scattered across the codebase.

  • InstallationAdapter wraps pykotor.extract.installation.Installation
    and converts pykotor-specific types into stable ResourceEntry objects.

  • PyKotorRegistryAdapter replaces the module-level INSTALLATIONS global
    dict (Common Coupling) with an object whose identity is explicit and
    injectable.

  • FileSystemModelLocator + InstallationModelLocator are Adapters for the
    ModelLocatorPort; they can be composed via CompositeModelLocator.
"""

from __future__ import annotations

import logging
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from kotormcp.ports import (
    AuditResult,
    InstallationPort,
    InstallationRegistryPort,
    ModelInfo,
    ModelLocatorPort,
    ModelParserPort,
    ResourceEntry,
)

log = logging.getLogger(__name__)

# ── pykotor availability guard ─────────────────────────────────────────────────

try:
    from pykotor.common.misc import Game
    from pykotor.extract.installation import Installation, SearchLocation
    from pykotor.resource.type import ResourceType
    from pykotor.tools.path import CaseAwarePath, find_kotor_paths_from_default
    _PYKOTOR_AVAILABLE = True
except ImportError:
    _PYKOTOR_AVAILABLE = False
    Game = None  # type: ignore[assignment, misc]
    Installation = None  # type: ignore[assignment, misc]
    SearchLocation = None  # type: ignore[assignment, misc]
    ResourceType = None  # type: ignore[assignment, misc]
    CaseAwarePath = None  # type: ignore[assignment, misc]

    def find_kotor_paths_from_default() -> dict:  # type: ignore[misc]
        return {}


# ── Default search order ───────────────────────────────────────────────────────

_DEFAULT_ORDER = ["OVERRIDE", "MODULES", "CHITIN"]


def _search_order(names: Optional[List[str]]) -> list:
    if not _PYKOTOR_AVAILABLE:
        return []
    order_names = names or _DEFAULT_ORDER
    result = []
    for n in order_names:
        try:
            result.append(getattr(SearchLocation, n.upper()))
        except AttributeError:
            pass
    return result or [SearchLocation.OVERRIDE, SearchLocation.MODULES, SearchLocation.CHITIN]


def _resource_data_bytes(resource: Any) -> bytes:
    data = resource.data() if callable(getattr(resource, "data", None)) else getattr(resource, "data", b"")
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if hasattr(data, "read"):
        return data.read()
    return bytes(data)


# ── InstallationAdapter ────────────────────────────────────────────────────────

class InstallationAdapter(InstallationPort):
    """
    Wraps a pykotor Installation object behind the InstallationPort contract.

    All knowledge of pykotor internals is contained here — callers only see
    the stable ResourceEntry / str / int types defined in ports.py.
    """

    def __init__(self, installation: "Installation", game: Any):
        self._inst = installation
        self._game = game

    def path(self) -> str:
        return str(self._inst.path())

    def game_name(self) -> str:
        return self._game.name if hasattr(self._game, "name") else str(self._game)

    def module_names(self) -> List[str]:
        try:
            return self._inst.modules_list() or []
        except Exception:
            return []

    def override_count(self) -> int:
        try:
            return sum(1 for _ in self._inst.override_resources())
        except Exception:
            return -1

    def iter_resources(
        self,
        location: str = "all",
        module_filter: Optional[str] = None,
    ) -> Iterator[ResourceEntry]:
        lowered = location.lower()
        if lowered == "auto":
            lowered = "all"

        if lowered in {"override", "all"}:
            try:
                for r in self._inst.override_resources():
                    entry = self._to_entry(r, "override")
                    if entry:
                        yield entry
            except Exception:
                pass

        if lowered in {"core", "all"}:
            try:
                for r in self._inst.core_resources():
                    entry = self._to_entry(r, "core")
                    if entry:
                        yield entry
            except Exception:
                pass

        if lowered.startswith("module:"):
            _, alias = lowered.split(":", 1)
            resolved = self._resolve_module_alias(alias)
            if resolved:
                try:
                    for r in self._inst.module_resources(resolved):
                        entry = self._to_entry(r, f"module:{resolved}")
                        if entry:
                            yield entry
                except Exception:
                    pass
            return

        if lowered in {"modules", "all"}:
            try:
                for mod_name in self._inst.modules_list():
                    if module_filter and module_filter.lower() not in mod_name.lower():
                        continue
                    try:
                        for r in self._inst.module_resources(mod_name):
                            entry = self._to_entry(r, f"module:{mod_name}")
                            if entry:
                                yield entry
                    except Exception:
                        pass
            except Exception:
                pass

        if lowered in {"chitin", "bif", "all"}:
            try:
                for r in self._inst.chitin_resources():
                    entry = self._to_entry(r, "chitin")
                    if entry:
                        yield entry
            except Exception:
                pass

    def get_resource(
        self,
        resref: str,
        restype: str,
        order: Optional[List[str]] = None,
    ) -> Optional[ResourceEntry]:
        if not _PYKOTOR_AVAILABLE:
            return None
        try:
            rt = ResourceType.from_extension(restype.lstrip(".").lower())
            if str(rt) == "INVALID":
                return None
            sl_order = _search_order(order)
            result = self._inst.resource(resref, rt, order=sl_order)
            if result is None:
                return None
            data = _resource_data_bytes(result)
            return ResourceEntry(
                resref=result.resname() if callable(result.resname) else result.resname,
                restype=rt.name,
                extension=rt.extension,
                size=len(data),
                source="installation",
                data=data,
            )
        except Exception as exc:
            log.debug("get_resource error for %s.%s: %s", resref, restype, exc)
            return None

    def get_texture_resource(self, resref: str) -> Optional[ResourceEntry]:
        """Resolve TPC/TGA bytes using the installation's texture-pack order."""
        if not _PYKOTOR_AVAILABLE:
            return None
        try:
            result, _txi = self._inst.texture_resource_result(resref)
            if result is None:
                return None
            restype = result.restype() if callable(getattr(result, "restype", None)) else result.restype
            data = _resource_data_bytes(result)
            return ResourceEntry(
                resref=(
                    result.resname()
                    if callable(getattr(result, "resname", None))
                    else str(getattr(result, "resname", resref))
                ),
                restype=str(getattr(restype, "name", restype) or ""),
                extension=str(getattr(restype, "extension", "") or "").lower(),
                size=len(data),
                source="installation_texture_priority",
                data=data,
            )
        except Exception as exc:
            log.debug("get_texture_resource error for %s: %s", resref, exc)
            return None


    def talktable_string(self, strref: int) -> str:
        return self._inst.talktable().string(strref)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _to_entry(self, resource: Any, source: str) -> Optional[ResourceEntry]:
        try:
            name = resource.resname().lower() if callable(resource.resname) else resource.resname.lower()
            rt = resource.restype() if callable(resource.restype) else resource.restype
            return ResourceEntry(
                resref=name,
                restype=rt.name,
                extension=rt.extension,
                size=resource.size() if callable(resource.size) else resource.size,
                source=source,
            )
        except Exception:
            return None

    def _resolve_module_alias(self, alias: str) -> Optional[str]:
        alias_lower = alias.lower()
        try:
            modules = self._inst.modules_list()
            lookup = {n.lower(): n for n in modules}
            if alias_lower in lookup:
                return lookup[alias_lower]
            for candidate in modules:
                if alias_lower in candidate.lower():
                    return candidate
        except Exception:
            pass
        return None


class InstallationTextureCache:
    """Read-only PIL-image provider backed by an InstallationPort."""

    def __init__(self, installation: InstallationPort) -> None:
        self._installation = installation
        self._requested: list[str] = []
        self._decoded: list[str] = []
        self._missing: list[str] = []
        self._errors: Dict[str, str] = {}

    def get(self, texture_name: str):
        name = Path(str(texture_name or "").strip()).stem
        if not name or name.upper() in {"NULL", "BLACK", "NONE"}:
            return None
        if name.lower() not in {item.lower() for item in self._requested}:
            self._requested.append(name)
        resource = self._installation.get_texture_resource(name)
        if resource is None or not resource.data:
            if name.lower() not in {item.lower() for item in self._missing}:
                self._missing.append(name)
            return None
        try:
            if resource.extension == "tga":
                from PIL import Image  # noqa: PLC0415

                with Image.open(BytesIO(resource.data)) as source:
                    image = source.convert("RGBA").copy()
            else:
                from src.core.game.kotor_loader import load_tpc_as_pil  # noqa: PLC0415

                image = load_tpc_as_pil(resource.data)
                if image is not None and getattr(image, "mode", "") != "RGBA":
                    image = image.convert("RGBA")
            if image is None:
                raise ValueError("decoder returned no image")
            if name.lower() not in {item.lower() for item in self._decoded}:
                self._decoded.append(name)
            return image
        except Exception as exc:
            self._errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    def summary(self) -> Dict[str, Any]:
        return {
            "requested": list(self._requested),
            "decoded": list(self._decoded),
            "missing": list(self._missing),
            "errors": dict(self._errors),
            "counts": {
                "requested": len(self._requested),
                "decoded": len(self._decoded),
                "missing": len(self._missing),
                "errors": len(self._errors),
            },
        }


# ── PyKotorRegistryAdapter ─────────────────────────────────────────────────────

# Game alias table (kept here — NOT in global state)
_GAME_ALIASES: Dict[str, Any] = {}
_ENV_HINTS: Dict[Any, Tuple[str, ...]] = {}
_DEFAULT_PATH_CACHE: Dict[Any, list] = {}

if _PYKOTOR_AVAILABLE:
    _GAME_ALIASES = {
        "k1": Game.K1,
        "kotori": Game.K1,
        "swkotor": Game.K1,
        "k2": Game.K2,
        "tsl": Game.K2,
        "kotor2": Game.K2,
    }
    _ENV_HINTS = {
        Game.K1: ("K1_PATH", "KOTOR_PATH", "KOTOR1_PATH"),
        Game.K2: ("K2_PATH", "TSL_PATH", "KOTOR2_PATH"),
    }
    try:
        _DEFAULT_PATH_CACHE = find_kotor_paths_from_default()
    except Exception:
        _DEFAULT_PATH_CACHE = {}


class PyKotorRegistryAdapter(InstallationRegistryPort):
    """
    Replaces the module-level INSTALLATIONS global (Common Coupling) with an
    explicit, injectable object that satisfies InstallationRegistryPort.

    Multiple independent instances can coexist — essential for test isolation.
    """

    def __init__(self) -> None:
        self._cache: Dict[Any, InstallationPort] = {}

    def resolve(self, label: Optional[str]) -> Optional[Any]:
        if not _PYKOTOR_AVAILABLE or label is None:
            return None
        return _GAME_ALIASES.get(label.strip().lower())

    def load(
        self,
        game: Any,
        explicit_path: Optional[str] = None,
    ) -> InstallationPort:
        if not _PYKOTOR_AVAILABLE:
            raise ImportError("pykotor is not installed. Install with: pip install pykotor>=2.3.1")

        cached = self._cache.get(game)
        if cached is not None:
            return cached

        for candidate in self._iter_candidates(game, explicit_path):
            if candidate.is_dir():
                adapter = InstallationAdapter(Installation(candidate), game)
                self._cache[game] = adapter
                return adapter

        hints = _ENV_HINTS.get(game, ())
        primary_env = hints[0] if hints else "K1_PATH"
        raise ValueError(
            f"Cannot locate installation for {game.name}. "
            f"Set {primary_env} or provide an explicit path."
        )

    def clear(self) -> None:
        self._cache.clear()

    def iter_candidate_paths(self, game: Any, explicit: Optional[str]) -> Iterator:
        """Public alias for _iter_candidates (used by installation tools)."""
        return self._iter_candidates(game, explicit)

    def default_paths(self, game: Any) -> list:
        """Return default search paths for a game (used by detectInstallations tool)."""
        return _DEFAULT_PATH_CACHE.get(game, [])

    def default_path_keys(self, game: Any) -> set:
        return {str(p).lower() for p in _DEFAULT_PATH_CACHE.get(game, [])}

    def _iter_candidates(self, game: Any, explicit: Optional[str]):
        seen: set = set()
        if explicit:
            candidate = CaseAwarePath(explicit).expanduser().resolve()
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                yield candidate
        for env_name in _ENV_HINTS.get(game, ()):
            env_value = os.environ.get(env_name)
            if env_value:
                candidate = CaseAwarePath(env_value).expanduser().resolve()
                key = str(candidate).lower()
                if key not in seen:
                    seen.add(key)
                    yield candidate
        for default_path in _DEFAULT_PATH_CACHE.get(game, []):
            key = str(default_path).lower()
            if key not in seen:
                seen.add(key)
                yield default_path


# ── Shared singleton registry (backward-compatible) ───────────────────────────

# One shared registry satisfies existing callers of state.py functions.
# New code should inject a registry explicitly.
_default_registry: Optional[PyKotorRegistryAdapter] = None


def get_default_registry() -> PyKotorRegistryAdapter:
    """Return the process-wide default registry (lazy-init)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = PyKotorRegistryAdapter()
    return _default_registry


# ── CompositeModelLocator ─────────────────────────────────────────────────────

class FileSystemModelLocator(ModelLocatorPort):
    """
    Adapter: locate MDL files on the local filesystem.

    Searches:
      1. Exact path (absolute or relative)
      2. Path + .mdl suffix
      3. Standard local game_data/ directory
    """

    def __init__(self, project_root: Optional[Path] = None):
        self._root = project_root or Path(__file__).parent.parent.parent

    def locate(
        self,
        resref: str,
        game_alias: Optional[str] = None,
        game_path: Optional[str] = None,
    ) -> Tuple[str, bytes, bytes]:
        p = Path(resref)
        candidates = [p, p.with_suffix(".mdl")]
        for c in candidates:
            if c.exists() and c.suffix.lower() in (".mdl", ""):
                mdl = c.read_bytes()
                mdx = self._load_mdx(c)
                return str(c), mdl, mdx

        # Local game_data directory
        for search_dir in [self._root / "game_data"]:
            candidate = search_dir / f"{resref}.mdl"
            if candidate.exists():
                mdl = candidate.read_bytes()
                mdx = self._load_mdx(candidate)
                return str(candidate), mdl, mdx

        raise FileNotFoundError(f"MDL not found on filesystem: {resref}")

    @staticmethod
    def _load_mdx(mdl_path: Path) -> bytes:
        mdx = mdl_path.with_suffix(".mdx")
        if mdx.exists():
            return mdx.read_bytes()
        # try same stem different case
        alt = mdl_path.parent / (mdl_path.stem + ".mdx")
        if alt.exists():
            return alt.read_bytes()
        return b""


class InstallationModelLocator(ModelLocatorPort):
    """
    Adapter: locate MDL resources inside a KotOR installation.
    """

    def __init__(self, registry: InstallationRegistryPort):
        self._registry = registry

    def locate(
        self,
        resref: str,
        game_alias: Optional[str] = None,
        game_path: Optional[str] = None,
    ) -> Tuple[str, bytes, bytes]:
        if not game_alias or not _PYKOTOR_AVAILABLE:
            raise FileNotFoundError("No game alias provided")

        game = self._registry.resolve(game_alias)
        if game is None:
            raise FileNotFoundError(f"Unknown game alias: {game_alias}")

        installation = self._registry.load(game, game_path)
        entry = installation.get_resource(resref, "mdl")
        if entry is None:
            raise FileNotFoundError(f"MDL not found in installation: {resref}")

        # A binary Odyssey model is a paired MDL + MDX resource.  The MDL owns
        # the hierarchy and mesh headers while the companion MDX owns the
        # vertex streams, including skin weights and bone-palette indices.  An
        # earlier MCP adapter returned only the MDL bytes, which made skinned
        # creatures appear structurally valid while silently dropping every
        # per-vertex influence before FBX export.
        mdx_entry = installation.get_resource(resref, "mdx")
        mdx_data = mdx_entry.data if mdx_entry is not None else b""
        return f"installation:{resref}.mdl", entry.data, mdx_data


class CompositeModelLocator(ModelLocatorPort):
    """
    Adapter: try multiple locators in order (filesystem first, then installation).

    This implements the Chain-of-Responsibility pattern described in
    Khononov's 'ports and adapters' case study — each adapter owns
    its own failure domain, composites orchestrate fallback.
    """

    def __init__(self, locators: List[ModelLocatorPort]):
        self._locators = locators

    def locate(
        self,
        resref: str,
        game_alias: Optional[str] = None,
        game_path: Optional[str] = None,
    ) -> Tuple[str, bytes, bytes]:
        errors = []
        for loc in self._locators:
            try:
                return loc.locate(resref, game_alias, game_path)
            except FileNotFoundError as exc:
                errors.append(str(exc))
        raise FileNotFoundError(
            f"MDL not found: {resref}. Tried: " + "; ".join(errors)
        )


# ── MDLBinaryParser adapter ────────────────────────────────────────────────────

class MDLBinaryParserAdapter(ModelParserPort):
    """Adapter: parse raw MDL+MDX bytes using PyKotor directly."""

    def __init__(self, src_dir: Optional[str] = None):
        self._src_dir = src_dir or str(Path(__file__).parent.parent.parent)

    def parse(self, mdl: bytes, mdx: bytes, path_label: str) -> Any:
        if self._src_dir not in sys.path:
            sys.path.insert(0, self._src_dir)
        from core.game.kotor_loader import load_model_from_bytes  # noqa: PLC0415
        return load_model_from_bytes(mdl, mdx)


# ── ModelAnalyzer ──────────────────────────────────────────────────────────────

class ModelAnalyzer:
    """
    Service: extract structured information from a parsed KotorModel.

    This class consolidates the 'how to interrogate a model' knowledge
    that was previously duplicated across handle_model_info() and
    handle_audit() (Connascence of Algorithm).  Both tools now call
    the same service, with the result delivered as stable data contracts
    (ModelInfo, AuditResult).
    """

    # ── Public API ─────────────────────────────────────────────────────────────

    def model_info(self, model: Any, resref: str, path: str) -> ModelInfo:
        """Build a ModelInfo contract from a parsed model."""
        nodes = self._all_nodes(model)
        mesh_nodes = self._mesh_nodes(model, nodes)
        bones = self._bone_nodes(model)
        bbox_min, bbox_max = self._bbox(model)
        total_verts = sum(
            len(n.vertices) for n in mesh_nodes
            if hasattr(n, "vertices") and n.vertices is not None
        )
        total_faces = sum(
            len(n.faces) for n in mesh_nodes
            if hasattr(n, "faces") and n.faces is not None
        )
        node_count = self._node_count(model, nodes)
        return ModelInfo(
            resref=resref,
            path=path,
            node_count=node_count,
            mesh_node_count=len(mesh_nodes),
            total_vertices=total_verts,
            total_faces=total_faces,
            bone_count=len(bones),
            bones=[b.name if hasattr(b, "name") else str(b) for b in bones[:30]],
            animations=[a.name for a in getattr(model, "animations", [])],
            bounding_box_min=list(bbox_min) if bbox_min is not None else None,
            bounding_box_max=list(bbox_max) if bbox_max is not None else None,
            classification=str(getattr(model, "classification", None)),
            supermodel=getattr(model, "supermodel", None),
        )

    def audit(self, model: Any, resref: str) -> AuditResult:
        """Build an AuditResult contract from a parsed model."""
        nodes = self._all_nodes(model)
        mesh_nodes = self._mesh_nodes(model, nodes)
        issues: List[str] = []
        warnings: List[str] = []

        for n in mesh_nodes:
            verts = n.vertices if hasattr(n, "vertices") and n.vertices is not None else []
            if hasattr(n, "uvs") and n.uvs is not None and len(n.uvs) != len(verts):
                issues.append(
                    f"Node '{n.name}': UV count mismatch ({len(n.uvs)} uvs vs {len(verts)} verts)"
                )
            if hasattr(n, "normals") and n.normals is not None and len(n.normals) != len(verts):
                warnings.append(f"Node '{n.name}': Normal count mismatch")
            if hasattr(n, "faces") and n.faces is not None and len(n.faces) == 0:
                warnings.append(f"Node '{n.name}': Has vertices but no faces")

        bbox_min, bbox_max = self._bbox(model)
        bbox_ok = (
            bbox_min is not None
            and bbox_max is not None
            and any(abs(a - b) > 0.001 for a, b in zip(bbox_min, bbox_max))
        )
        node_count = self._node_count(model, nodes)

        return AuditResult(
            resref=resref,
            status="ok" if not issues else "issues_found",
            node_count=node_count,
            mesh_node_count=len(mesh_nodes),
            bounding_box_ok=bbox_ok,
            issues=issues,
            warnings=warnings,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _all_nodes(model: Any) -> list:
        if callable(getattr(model, "all_nodes", None)):
            return list(model.all_nodes())
        if hasattr(model, "all_nodes"):
            return list(model.all_nodes)
        return []

    @staticmethod
    def _mesh_nodes(model: Any, all_nodes: list) -> list:
        # Include both mesh nodes (trimesh/dangly/saber) and skin nodes since
        # both hold renderable geometry.  KotOR skin nodes have is_skin=True but
        # is_mesh=False, yet they have vertex data identical to a trimesh.
        if callable(getattr(model, "mesh_nodes", None)):
            mesh = list(model.mesh_nodes())
        elif hasattr(model, "mesh_nodes"):
            mesh = list(model.mesh_nodes)
        else:
            mesh = []
        # Add skin nodes not already in mesh list
        mesh_set = {id(n) for n in mesh}
        for n in all_nodes:
            if id(n) not in mesh_set and getattr(n, "is_skin", False):
                mesh.append(n)
                mesh_set.add(id(n))
        if not mesh:
            # Final fallback: any node with vertices
            return [
                n for n in all_nodes
                if hasattr(n, "vertices") and n.vertices is not None and len(n.vertices) > 0
            ]
        return mesh

    @staticmethod
    def _bone_nodes(model: Any) -> list:
        raw = model.bone_nodes() if callable(getattr(model, "bone_nodes", None)) else (
            model.bone_nodes if hasattr(model, "bone_nodes") else []
        )
        return list(raw) if raw else []

    @staticmethod
    def _bbox(model: Any):
        bbox_min = getattr(model, "bb_min", None) or getattr(model, "bounding_box_min", None)
        bbox_max = getattr(model, "bb_max", None) or getattr(model, "bounding_box_max", None)
        return bbox_min, bbox_max

    @staticmethod
    def _node_count(model: Any, all_nodes: list) -> int:
        if callable(getattr(model, "node_count", None)):
            return model.node_count()
        if hasattr(model, "node_count"):
            return model.node_count
        return len(all_nodes)
