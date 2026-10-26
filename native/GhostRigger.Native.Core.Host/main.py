#!/usr/bin/env python3
"""Native Visual Studio entrypoint for GhostRigger.

This file is owned by `GhostRigger.Native.Core.Host` and is copied beside
`GhostStudio.exe` during the Visual Studio build. It intentionally does not
import or execute the repository-root `main.py`.
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import importlib.abc
import importlib.machinery
import logging
import os
from pathlib import Path
import sys
import traceback


sys.dont_write_bytecode = True
_HOST_DIR = Path(__file__).resolve().parent


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "GhostRigger.sln").exists() or (
        (path / "pyproject.toml").exists() and (path / "native").is_dir()
    )


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if _looks_like_repo_root(candidate):
            return candidate.resolve()
    return start.resolve()


def _payload_has_python_sources(payload_root: Path) -> bool:
    source_root = payload_root / "src"
    if not source_root.is_dir():
        return False
    try:
        next(source_root.rglob("*.py"))
    except StopIteration:
        return False
    return True


def _source_package_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    if (repo_root / "src").is_dir():
        roots.append(repo_root)
    native_root = repo_root / "native"
    if native_root.is_dir():
        for project_dir in sorted(native_root.glob("GhostRigger*")):
            python_root = project_dir / "Python"
            if (python_root / "src").is_dir():
                roots.append(python_root)
    return roots


_REPO_ROOT = Path(os.environ.get("GHOSTRIGGER_NATIVE_REPO_ROOT", "") or _find_repo_root(_HOST_DIR)).resolve()


class _DllPythonPayloadImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, build_dir: Path, asset_root: Path) -> None:
        import ctypes

        self._ctypes = ctypes
        self._build_dir = Path(build_dir)
        self._asset_root = Path(asset_root)
        self._modules: dict[str, dict[str, object]] = {}
        self._packages: set[str] = set()
        self._namespace_packages: dict[str, str] = {}
        self._dlls: list[object] = []
        self._dll_directory_cookie = None
        if hasattr(os, "add_dll_directory") and self._build_dir.is_dir():
            try:
                self._dll_directory_cookie = os.add_dll_directory(str(self._build_dir))
            except OSError:
                self._dll_directory_cookie = None
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.FindResourceA.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        self._kernel32.FindResourceA.restype = ctypes.c_void_p
        self._kernel32.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._kernel32.LoadResource.restype = ctypes.c_void_p
        self._kernel32.LockResource.argtypes = [ctypes.c_void_p]
        self._kernel32.LockResource.restype = ctypes.c_void_p
        self._kernel32.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._kernel32.SizeofResource.restype = ctypes.c_uint32
        self._load_manifests()

    def _load_manifests(self) -> None:
        import json

        for dll_path in self._build_dir.glob("*.dll"):
            try:
                dll = self._ctypes.CDLL(str(dll_path), winmode=8)
                manifest_fn = dll.gr_python_payload_manifest_json
                manifest_fn.restype = self._ctypes.c_char_p
                manifest = json.loads((manifest_fn() or b"{}").decode("utf-8-sig"))
            except Exception:
                continue
            self._dlls.append(dll)
            for row in manifest.get("files", []) or []:
                resource_name = str(row.get("resource_name") or "")
                packaged_path = str(row.get("packaged_path") or "").replace("\\", "/")
                if not resource_name or not packaged_path.endswith(".py"):
                    continue
                if packaged_path.startswith("Python/"):
                    packaged_path = packaged_path[len("Python/"):]
                parts = [part for part in packaged_path.split("/") if part]
                if not parts or ".." in parts:
                    continue
                module_name, is_package = self._module_name_for_parts(parts)
                if not module_name or module_name in self._modules:
                    continue
                filename = self._asset_root.joinpath(*parts)
                self._modules[module_name] = {
                    "dll": dll,
                    "resource_name": resource_name.encode("ascii", errors="ignore"),
                    "sha256": str(row.get("sha256") or ""),
                    "filename": str(filename),
                    "package_dir": str(filename.parent),
                    "is_package": is_package,
                }
                if is_package:
                    self._packages.add(module_name)
                self._register_namespace_parents(parts, module_name)

    @staticmethod
    def _module_name_for_parts(parts: list[str]) -> tuple[str, bool]:
        if parts[-1] == "__init__.py":
            names = parts[:-1]
            is_package = True
        else:
            names = parts[:-1] + [parts[-1][:-3]]
            is_package = False
        if not names:
            return "", False
        return ".".join(names), is_package

    def _register_namespace_parents(self, parts: list[str], module_name: str) -> None:
        package_parts = parts[:-1] if parts[-1] != "__init__.py" else parts[:-2]
        for index in range(1, len(package_parts) + 1):
            package_name = ".".join(package_parts[:index])
            if not package_name or package_name in self._modules:
                continue
            package_dir = self._asset_root.joinpath(*package_parts[:index])
            self._namespace_packages.setdefault(package_name, str(package_dir))

    @property
    def module_count(self) -> int:
        return len(self._modules)

    def find_spec(self, fullname: str, path=None, target=None):
        record = self._modules.get(fullname)
        if record is None:
            package_dir = self._namespace_packages.get(fullname)
            if package_dir is None:
                return None
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [package_dir]
            return spec
        spec = importlib.machinery.ModuleSpec(
            fullname,
            self,
            origin=str(record["filename"]),
            is_package=bool(record["is_package"]),
        )
        if bool(record["is_package"]):
            spec.submodule_search_locations = [str(record["package_dir"])]
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        import linecache

        fullname = module.__spec__.name
        record = self._modules[fullname]
        filename = str(record["filename"])
        source = self.get_source(fullname)
        code = compile(source, filename, "exec")
        module.__file__ = filename
        module.__loader__ = self
        if bool(record["is_package"]):
            module.__path__ = [str(record["package_dir"])]
            module.__package__ = fullname
        else:
            module.__package__ = fullname.rpartition(".")[0]
        lines = source.splitlines(keepends=True)
        linecache.cache[filename] = (len(source), None, lines, filename)
        exec(code, module.__dict__)

    def get_filename(self, fullname: str) -> str:
        return str(self._modules[fullname]["filename"])

    def get_source(self, fullname: str) -> str:
        data = self.get_data(self.get_filename(fullname), fullname=fullname)
        return data.decode("utf-8-sig")

    def get_data(self, path: str, *, fullname: str = "") -> bytes:
        record = self._modules[fullname] if fullname else self._record_for_path(path)
        resource = self._kernel32.FindResourceA(record["dll"]._handle, record["resource_name"], self._ctypes.c_void_p(10))
        if not resource:
            if record.get("sha256") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
                return b""
            raise FileNotFoundError(path)
        handle = self._kernel32.LoadResource(record["dll"]._handle, resource)
        data_ptr = self._kernel32.LockResource(handle) if handle else None
        size = int(self._kernel32.SizeofResource(record["dll"]._handle, resource))
        if not data_ptr or size <= 0:
            raise FileNotFoundError(path)
        return self._ctypes.string_at(data_ptr, size)

    def _record_for_path(self, path: str) -> dict[str, object]:
        normalized = str(Path(path))
        for record in self._modules.values():
            if str(record["filename"]) == normalized:
                return record
        raise FileNotFoundError(path)


def _install_native_python_payload_importer() -> Path | None:
    if os.name != "nt":
        return None
    try:
        import ctypes  # noqa: F401
    except Exception:
        return None

    build_dir = Path(os.environ.get("GHOSTRIGGER_NATIVE_BUILD_OUTPUT_DIR", "") or _HOST_DIR)
    payload_root = Path(os.environ.get("GHOSTRIGGER_NATIVE_PAYLOAD_ROOT", "") or (build_dir / "GhostRiggerPythonPayload"))
    importer = _DllPythonPayloadImporter(build_dir, payload_root)
    if importer.module_count <= 0:
        return None
    if not any(isinstance(item, _DllPythonPayloadImporter) for item in sys.meta_path):
        sys.meta_path.insert(0, importer)
    os.environ["GHOSTRIGGER_NATIVE_PAYLOAD_ROOT"] = str(payload_root)
    return payload_root


def _legacy_extracted_payload_root() -> Path | None:
    payload_root = Path(os.environ.get("GHOSTRIGGER_NATIVE_PAYLOAD_ROOT", "") or (_HOST_DIR / "GhostRiggerPythonPayload"))
    if payload_root.is_dir() and _payload_has_python_sources(payload_root):
        return payload_root
    return None


_NATIVE_PAYLOAD_ROOT: Path | None = None
_ENV_PAYLOAD_ROOT = os.environ.get("GHOSTRIGGER_NATIVE_PAYLOAD_ROOT", "").strip()
_DLL_PAYLOAD_ROOT = _install_native_python_payload_importer()
if _DLL_PAYLOAD_ROOT is not None:
    _NATIVE_PAYLOAD_ROOT = _DLL_PAYLOAD_ROOT
else:
    _NATIVE_PAYLOAD_ROOT = _legacy_extracted_payload_root()
if _NATIVE_PAYLOAD_ROOT is None:
    for _source_root in reversed(_source_package_roots(_REPO_ROOT)):
        _source_root_str = str(_source_root)
        if _source_root_str not in sys.path:
            sys.path.insert(0, _source_root_str)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_APP_ROOT = (_NATIVE_PAYLOAD_ROOT if _NATIVE_PAYLOAD_ROOT is not None and _NATIVE_PAYLOAD_ROOT.is_dir() else _REPO_ROOT).resolve()
os.environ["GHOSTRIGGER_NATIVE_APP_ROOT"] = str(_APP_ROOT)
try:
    os.chdir(_APP_ROOT)
except OSError:
    pass
_LOG_DIR = _APP_ROOT / "Logs"
_CURRENT_LOGFILE: str | None = None


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "debug"}


def _setup_logging() -> str | None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    for entry in _LOG_DIR.glob("*.log"):
        try:
            entry.unlink()
        except OSError:
            pass

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    logfile = _LOG_DIR / f"ghostrigger_{stamp}.log"
    level = logging.DEBUG if _env_enabled("GHOSTRIGGER_DEBUG_LOG") else logging.INFO
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    try:
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s", datefmt="%H:%M:%S"))
        root_logger.addHandler(file_handler)
    except OSError:
        logfile = None

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)-34.34s  %(message)s", datefmt="%H:%M:%S"))
        root_logger.addHandler(stream_handler)

    global _CURRENT_LOGFILE
    _CURRENT_LOGFILE = str(logfile) if logfile is not None else None
    if _CURRENT_LOGFILE:
        os.environ["GHOSTRIGGER_CURRENT_LOGFILE"] = _CURRENT_LOGFILE
    return _CURRENT_LOGFILE


def _flush_all_handlers() -> None:
    for handler in list(logging.getLogger().handlers):
        try:
            handler.flush()
        except Exception:
            pass


def _install_exception_hooks(logfile: str | None) -> None:
    crash_log = logging.getLogger("ghostrigger.crash")

    def _handle_uncaught(exc_type, exc_value, exc_tb) -> None:
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        crash_log.critical("UNHANDLED EXCEPTION:\n%s", message)
        _flush_all_handlers()
        print(f"\n{'=' * 60}\nGhostStudio CRASH - see Logs/ for full trace\n{'=' * 60}\n{message}", file=sys.stderr)

    sys.excepthook = _handle_uncaught


def _install_atexit_flush() -> None:
    shutdown_log = logging.getLogger("ghostrigger.shutdown")

    def _atexit_flush() -> None:
        shutdown_log.info("GhostRigger native-host atexit flush.")
        _flush_all_handlers()

    atexit.register(_atexit_flush)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GhostStudio native host")
    parser.add_argument("--gui", default="qt")
    parser.add_argument("--unreal-snapshot", help="Open the native Unreal import workflow.")
    parser.add_argument("model", nargs="?", help="Optional startup model path.")
    parser.add_argument("--tga", action="append", default=[])
    parser.add_argument("--texture", dest="tga", action="append")
    parser.add_argument("--texture-dir")
    parser.add_argument("--game", choices=("K1", "K2", "k1", "k2"))
    return parser.parse_args(argv)


def _precache_themes(app_dir: Path, log: logging.Logger) -> None:
    from src.gui.libtheme.theme_applier import ThemeApplier
    from src.gui.libtheme.theme_loader import ThemeLoader
    from src.gui.libtheme.theme_settings import user_config_root

    loader = ThemeLoader()
    themes = dict(loader.load_dir(app_dir / "config" / "themes" / "themes"))
    themes.update(loader.load_dir(user_config_root() / "themes"))
    if not themes:
        log.warning("Theme precache skipped; no theme XML files found.")
        return
    result = ThemeApplier.precache_stylesheets(sorted(themes.values(), key=lambda theme: (theme.id != "default", theme.name.lower(), theme.id)))
    log.info(
        "Theme precache complete: %d built, %d cached, %d failed in %.1f ms.",
        int(result["built"]),
        int(result["cached"]),
        int(result["failed"]),
        float(result["total_ms"]),
    )


def main(argv: list[str] | None = None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logfile = _setup_logging()
    log = logging.getLogger("ghostrigger.native_main")

    log.info("=" * 60)
    log.info("GhostStudio Native Host starting - Python %s", sys.version.split()[0])
    log.info("Native host entrypoint: %s", Path(__file__).resolve())
    log.info("Visual Studio build output: %s", os.environ.get("GHOSTRIGGER_NATIVE_BUILD_OUTPUT_DIR", ""))
    log.info("Repository root: %s", _REPO_ROOT)
    log.info("Native app root: %s", _APP_ROOT)
    log.info("Session log: %s", logfile or "DISABLED")
    log.info("=" * 60)

    if (args.gui or "qt").strip().lower() != "qt":
        log.warning("Only Qt is supported by the native host; continuing with Qt.")

    _install_exception_hooks(logfile)
    _install_atexit_flush()

    try:
        from src.core.qt_core.diagnostics.diagnostics import log_session_start
        log_session_start(str(_APP_ROOT), logfile or "(no log file)")
    except Exception as exc:
        log.debug("diagnostics.log_session_start failed: %s", exc)

    try:
        try:
            _precache_themes(_APP_ROOT, log)
        except Exception as exc:
            log.warning("Theme precache skipped after an unexpected error: %s", exc, exc_info=True)

        from src.gui.qt_lib.windows.qt_main_window import run as run_qt

        log.info("Qt launcher starting from native host entrypoint.")
        rc = run_qt(str(_APP_ROOT), startup_input=vars(args))
        log.info("Qt main window exited cleanly.")
        _flush_all_handlers()
        return rc
    except Exception:
        log.critical("Fatal error during Qt startup:\n%s", traceback.format_exc())
        _flush_all_handlers()
        raise


if __name__ == "__main__":
    main()
