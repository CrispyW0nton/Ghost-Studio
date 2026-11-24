"""Build Ghost Studio (Release x64) and package the Windows release zip.

Usage:
    python scripts/release/build_release.py --version 5.2.0 [--skip-build] [--msbuild PATH]

Output: build/release/GhostStudio-v<version>-win64.zip plus a .sha256 file.

Run it with 64-bit Python 3.13. The zip carries a private copy of that Python
in <zip>/python with the runtime packages from requirements.txt, so users just
unzip and start GhostStudio.exe (no Python install, no setup step).

The zip also carries <zip>/mcp: the Python sources and start scripts the MCP
servers (KotorMCP and friends) need when an AI agent launches them with the
bundled interpreter: <zip>/python/python.exe <zip>/mcp/scripts/mcp/start_kotormcp_stdio.py

MSBuild runs with a short PATH. A long user PATH plus the Visual C++ tool
folders overflows, and the post-build steps then fail with exit 9009
("xcopy" / "powershell" is not recognized). The payload generator gets the
full path of the Python running this script.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
REPO = KIT.parents[1]
OUT = REPO / "build" / "vs" / "x64" / "Release"
DIST = REPO / "build" / "release"
SKIP_SUFFIX = {".pdb", ".lib", ".exp", ".ilk", ".iobj", ".ipdb", ".log"}
STAGED_DIRS = {"scripts", "GhostRiggerPythonPayload"}
REPO_FILES = ("LICENSE", "NOTICE")
# Build-only packages from requirements.txt; they stay out of the zip.
BUILD_ONLY_REQS = {"pyinstaller", "sip"}
RUNTIME_FILES = ("python.exe", "pythonw.exe", "python313.dll", "python3.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt")
# Standard-library parts Ghost Studio never uses (no Tk GUI, no test suite).
LIB_SKIP = {"site-packages", "test", "idlelib", "tkinter", "turtledemo", "ensurepip"}
DLLS_SKIP = {"_tkinter.pyd", "tcl86t.dll", "tk86t.dll"}
SMOKE_IMPORTS = "import PySide6.QtWidgets, numpy, PIL, moderngl, wgpu, pykotor, trimesh, flask, mcp"
# MCP payload: the repo's import roots, copied as-is so scripts/mcp/*.py resolve <zip>/mcp as their root.
MCP_SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "test", "*.blend", "*.blend1")
MCP_SMOKE = (
    "import runpy, sys; m = runpy.run_path(sys.argv[1], run_name='kja_check'); "
    "from pathlib import Path; root = Path(sys.argv[1]).resolve().parents[2]; "
    "[sys.path.insert(0, str(p)) for p in reversed(m['_python_roots'](root))]; "
    "import kotormcp.server as s; assert s._MCP_AVAILABLE, 'mcp SDK missing'; s._build_mcp_server(); "
    "from kotormcp.tools import get_all_tools; print('kotormcp tools:', len(get_all_tools()))"
)


def find_msbuild(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get("MSBUILD"):
        return Path(os.environ["MSBUILD"])
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        found = subprocess.run(
            [str(vswhere), "-latest", "-requires", "Microsoft.Component.MSBuild",
             "-find", r"MSBuild\**\Bin\MSBuild.exe"],
            capture_output=True, text=True).stdout.strip().splitlines()
        if found:
            return Path(found[0])
    sys.exit("MSBuild not found; pass --msbuild or set MSBUILD")


def build_env() -> dict[str, str]:
    env = dict(os.environ)
    root = env.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([
        str(Path(sys.executable).parent), rf"{root}\System32", root,
        rf"{root}\System32\WindowsPowerShell\v1.0", rf"{root}\System32\Wbem"])
    return env


def build(msbuild: Path) -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    log = DIST / "msbuild_release.log"
    cmd = [str(msbuild), str(REPO / "GhostRigger.sln"), "/p:Configuration=Release", "/p:Platform=x64",
           f"/p:GhostRiggerPythonPayloadPython={sys.executable}",
           "/m:2", "/nr:false", "/nologo", "/v:minimal", f"/flp:LogFile={log};Verbosity=normal"]
    print("building:", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=REPO, env=build_env()).returncode:
        sys.exit(f"MSBuild failed; see {log}")


def stage(version: str) -> Path:
    exe = OUT / "GhostStudio.exe"
    if not exe.is_file():
        sys.exit(f"missing {exe}; build first")
    stage_dir = DIST / f"GhostStudio-v{version}-win64"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    for item in OUT.iterdir():
        if item.is_file() and item.suffix.lower() not in SKIP_SUFFIX:
            shutil.copy2(item, stage_dir / item.name)
        elif item.is_dir() and item.name in STAGED_DIRS:
            shutil.copytree(item, stage_dir / item.name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in REPO_FILES:
        shutil.copy2(REPO / name, stage_dir / name)
    bundle_python(stage_dir)
    stage_mcp(stage_dir)
    readme = (KIT / "README-FIRST.txt").read_text(encoding="utf-8").replace("@VERSION@", version)
    (stage_dir / "README-FIRST.txt").write_text(readme, encoding="utf-8", newline="\r\n")
    missing = [n for n in ("GhostStudio.exe", "python313.dll", "python/python.exe",
                           "python/python313.dll", "python/Lib/site-packages/PySide6")
               if not (stage_dir / n).exists()]
    missing += [n for n in ("mcp/scripts/mcp/start_kotormcp_stdio.py",
                            "mcp/native/GhostRigger.Core.Automation/Python/src/kotormcp/server.py")
                if not (stage_dir / n).exists()]
    if missing:
        sys.exit(f"stage is missing {missing}")
    print(f"staged {len(list(stage_dir.glob('GhostRigger*.dll')))} GhostRigger DLLs")
    return stage_dir


def runtime_requirements() -> Path:
    """requirements.txt minus build-only packages."""
    keep = []
    for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if not spec:
            continue
        name = spec.replace("_", "-").lower()
        for sep in "[<>=!~ ;":
            name = name.split(sep, 1)[0]
        if name not in BUILD_ONLY_REQS:
            keep.append(spec)
    reqs = DIST / "runtime-requirements.txt"
    reqs.write_text("\n".join(keep) + "\n", encoding="utf-8")
    return reqs


def lock_constraints() -> Path:
    """Pin every package to its uv.lock version, so the zip ships the versions
    the project is tested with (mcp 2.x, for example, removed the low-level
    Server.list_tools decorator KotorMCP uses). A package the lock resolves
    per Python version is pinned to the entry whose markers match this
    interpreter, which is the one copied into the zip."""
    import tomllib
    try:
        from packaging.markers import Marker
    except ImportError:
        from pip._vendor.packaging.markers import Marker
    # requirements.txt's own exact pins win (pykotor==2.3.1 is pinned on purpose).
    exact = {line.split("==", 1)[0].strip().replace("_", "-").lower()
             for line in (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if "==" in line.split("#", 1)[0]}
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    pins = set()
    for pkg in lock.get("package", []):
        if not (pkg.get("version") and pkg.get("source", {}).get("registry")):
            continue
        if pkg["name"].replace("_", "-").lower() in exact:
            continue
        markers = pkg.get("resolution-markers")
        if markers and not any(Marker(m).evaluate() for m in markers):
            continue
        pins.add(f"{pkg['name']}=={pkg['version']}")
    pins = sorted(pins)
    constraints = DIST / "runtime-constraints.txt"
    constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
    return constraints


def bundle_python(stage_dir: Path) -> None:
    """Copy the build Python 3.13 into <stage>/python and pip-install the
    runtime packages into its Lib/site-packages."""
    if sys.version_info[:2] != (3, 13) or sys.maxsize <= 2**32:
        sys.exit("run build_release.py with 64-bit Python 3.13; its runtime is bundled")
    base = Path(sys.base_prefix)
    dest = stage_dir / "python"
    dest.mkdir()
    for name in RUNTIME_FILES:
        shutil.copy2(base / name, dest / name)
    shutil.copytree(base / "DLLs", dest / "DLLs",
                    ignore=lambda _d, names: [n for n in names if n in DLLS_SKIP or n.endswith(".pdb")])
    shutil.copytree(base / "Lib", dest / "Lib",
                    ignore=lambda d, names: [n for n in names if n in {"__pycache__"} or n.endswith(".pyc")
                                             or (Path(d) == base / "Lib" and n in LIB_SKIP)])
    site = dest / "Lib" / "site-packages"
    reqs = runtime_requirements()
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-warn-script-location",
           "--target", str(site), "-r", str(reqs), "-c", str(lock_constraints())]
    print("bundling packages:", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode:
        sys.exit("pip install into the bundled runtime failed")
    shutil.rmtree(site / "bin", ignore_errors=True)  # console scripts point at the build Python
    # Precompile once; the host runs with write_bytecode off, and unchecked-hash
    # pycs stay valid after the zip rounds file times to 2 seconds.
    subprocess.run([sys.executable, "-m", "compileall", "-q", "-f", "-j", "0",
                    "--invalidation-mode", "unchecked-hash", str(dest / "Lib")])
    # GhostStudio.exe links python313.dll from its own folder; keep that copy
    # byte-identical to the runtime's so only one build of it ever loads.
    for name in ("python313.dll", "python3.dll"):
        shutil.copy2(dest / name, stage_dir / name)
    check = subprocess.run([str(dest / "python.exe"), "-I", "-c", SMOKE_IMPORTS], cwd=dest)
    if check.returncode:
        sys.exit("bundled runtime failed its import check")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"bundled Python {sys.version.split()[0]} runtime: {size / 1e6:.0f} MB", flush=True)


def stage_mcp(stage_dir: Path) -> None:
    """Copy the MCP servers' Python sources into <stage>/mcp, laid out like the
    repo, and check that KotorMCP imports and registers its tools with the
    bundled runtime."""
    dest = stage_dir / "mcp"
    for project in sorted((REPO / "native").glob("GhostRigger.*")):
        if (project / "Python").is_dir():
            shutil.copytree(project / "Python", dest / "native" / project.name / "Python", ignore=MCP_SKIP)
    if (REPO / "src").is_dir():
        shutil.copytree(REPO / "src", dest / "src", ignore=MCP_SKIP)
    (dest / "scripts" / "mcp").mkdir(parents=True)
    for script in (REPO / "scripts" / "mcp").glob("*.py"):
        shutil.copy2(script, dest / "scripts" / "mcp" / script.name)
    start = dest / "scripts" / "mcp" / "start_kotormcp_stdio.py"
    check = subprocess.run([str(stage_dir / "python" / "python.exe"), "-I", "-c", MCP_SMOKE, str(start)],
                           cwd=stage_dir, capture_output=True, text=True, timeout=600)
    print(check.stdout.strip()[-400:], flush=True)
    if check.returncode:
        print(check.stderr[-2000:], file=sys.stderr)
        sys.exit("KotorMCP failed its import check with the bundled runtime")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"MCP sources: {size / 1e6:.0f} MB", flush=True)


def pack(stage_dir: Path) -> None:
    archive_path = DIST / f"{stage_dir.name}.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path(stage_dir.name) / path.relative_to(stage_dir))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (DIST / (archive_path.name + ".sha256")).write_text(f"{digest}  {archive_path.name}\n", encoding="ascii")
    print(f"zip {archive_path} {archive_path.stat().st_size / 1e6:.1f} MB sha256 {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="release version, e.g. 5.2.0")
    parser.add_argument("--skip-build", action="store_true", help="package the existing Release build")
    parser.add_argument("--msbuild", help="path to MSBuild.exe")
    args = parser.parse_args()
    if not args.skip_build:
        build(find_msbuild(args.msbuild))
    pack(stage(args.version))


if __name__ == "__main__":
    main()
