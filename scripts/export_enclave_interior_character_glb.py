"""Export canonical enclave BAS characters as rigged GLB interchange sources."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAME_DEFAULT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
EXPORT_DEFAULT = Path(r"C:\UE\Enclave\Saved\MCPStudio\exports\enclave_characters_v1")


def _bootstrap() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "character"


async def _export(
    rows: list[dict],
    *,
    game_root: Path,
    export_root: Path,
    only: set[str],
) -> dict:
    from kotormcp.tools import handle_tool

    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(str(row["utc_resref"]).casefold(), row)
    if only:
        unique = {key: row for key, row in unique.items() if key in only}

    exports: list[dict] = []
    total = len(unique)
    for index, row in enumerate(unique.values(), start=1):
        utc_resref = str(row["utc_resref"])
        destination = export_root / _safe(utc_resref) / "glb_source"
        output_name = f"SK_{_safe(utc_resref)}_BAS"
        arguments = {
            "game": "k1",
            "game_path": str(game_root),
            "resref": row["body_resref"],
            "body_texture": row.get("body_texture", ""),
            "export_dir": str(destination),
            "output_name": output_name,
            "format": "glb",
            "export_rigging": True,
            "export_textures": True,
            "animation_names": list(row["animation_names"]),
        }
        if row.get("head_resref"):
            arguments["head_resref"] = row["head_resref"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            response = await handle_tool("ghostrigger_export_model_for_unreal", arguments)
        payload = json.loads(response["text"])
        asset = Path(str(payload.get("asset", "")))
        selected = [str(name).casefold() for name in payload.get("animations", [])]
        required = [str(name).casefold() for name in row["animation_names"]]
        if payload.get("status") != "ok" or not asset.is_file() or asset.stat().st_size < 1024:
            raise RuntimeError(f"{utc_resref} GLB export failed: {payload}")
        if not set(required).issubset(selected):
            raise RuntimeError(f"{utc_resref} GLB is missing required animations: {payload}")
        if row.get("head_resref") and payload.get("bas", {}).get("head_resref", "").casefold() != str(row["head_resref"]).casefold():
            raise RuntimeError(f"{utc_resref} GLB is missing BAS head evidence")
        exports.append({
            "utc_resref": utc_resref,
            "asset": str(asset),
            "bytes": asset.stat().st_size,
            "body_resref": row["body_resref"],
            "head_resref": row.get("head_resref", ""),
            "body_texture": row.get("body_texture", ""),
            "animations": payload.get("animations", []),
            "compatibility_profile": payload.get("compatibility_profile", ""),
        })
        print(f"[{index}/{total}] {utc_resref}", flush=True)

    return {
        "schema": "ghoststudio.enclave-interior-character-glb.v1",
        "module": "m13aa",
        "character_count": len(unique),
        "all_exports_valid": len(exports) == total,
        "exports": exports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=GAME_DEFAULT)
    parser.add_argument("--export-root", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    _bootstrap()
    roster_path = args.roster or args.export_root / "enclave_interior_roster_v1.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))["creatures"]
    receipt = asyncio.run(_export(
        roster,
        game_root=args.game_root,
        export_root=args.export_root,
        only={str(value).casefold() for value in args.only},
    ))
    suffix = "_" + "_".join(sorted({str(value).casefold() for value in args.only})) if args.only else ""
    receipt_path = args.export_root / f"enclave_interior_character_glb_v1{suffix}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path),
        "character_count": receipt["character_count"],
        "all_exports_valid": receipt["all_exports_valid"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
