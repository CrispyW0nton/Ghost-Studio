"""Build Ghost Studio (Release x64) and package the Windows release zip.

Usage:
    python scripts/release/build_release.py --version 5.2.0 [--skip-build] [--msbuild PATH]

Output: build/release/GhostStudio-v<version>-win64.zip plus a .sha256 file.

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
REPO_FILES = ("LICENSE", "NOTICE", "requirements.txt")


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
    shutil.copy2(KIT / "setup.bat", stage_dir / "setup.bat")
    readme = (KIT / "README-FIRST.txt").read_text(encoding="utf-8").replace("@VERSION@", version)
    (stage_dir / "README-FIRST.txt").write_text(readme, encoding="utf-8", newline="\r\n")
    missing = [n for n in ("GhostStudio.exe", "python313.dll") if not (stage_dir / n).exists()]
    if missing:
        sys.exit(f"stage is missing {missing}")
    print(f"staged {len(list(stage_dir.glob('GhostRigger*.dll')))} GhostRigger DLLs")
    return stage_dir


def pack(stage_dir: Path) -> None:
    archive_path = stage_dir.with_suffix(".zip")
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
