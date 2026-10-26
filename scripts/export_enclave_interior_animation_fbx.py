"""Export one Unreal-importable FBX per canonical enclave character animation."""

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
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "animation"


async def _export(
    rows: list[dict],
    *,
    game_root: Path,
    export_root: Path,
    only: set[str],
    selected_animations: set[str],
) -> dict:
    from kotormcp.tools import handle_tool

    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(str(row["utc_resref"]).casefold(), row)
    if only:
        unique = {key: row for key, row in unique.items() if key in only}
    exports: list[dict] = []
    requested_by_character = {
        key: (sorted(selected_animations) if selected_animations else list(row["animation_names"]))
        for key, row in unique.items()
    }
    total = sum(len(names) for names in requested_by_character.values())
    completed = 0
    for key, row in unique.items():
        utc_resref = str(row["utc_resref"])
        character_root = export_root / _safe(utc_resref)
        for animation_name in requested_by_character[key]:
            completed += 1
            destination = character_root / "animation_sources" / _safe(animation_name)
            output_name = f"A_{_safe(utc_resref)}_{_safe(animation_name)}"
            arguments = {
                "game": "k1",
                "game_path": str(game_root),
                "resref": row["body_resref"],
                "body_texture": row.get("body_texture", ""),
                "export_dir": str(destination),
                "output_name": output_name,
                "format": "glb",
                "export_rigging": True,
                # The Blender normalizer validates and rebuilds the complete
                # BAS material/UV presentation even for an animation-only
                # take.  Keep texture sidecars available so a targeted native
                # clip export cannot fail or silently emit an untextured base.
                "export_textures": True,
                "animation_names": [animation_name],
            }
            if row.get("head_resref"):
                arguments["head_resref"] = row["head_resref"]
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                response = await handle_tool("ghostrigger_export_model_for_unreal", arguments)
            payload = json.loads(response["text"])
            asset = Path(str(payload.get("asset", "")))
            animation_evidence = payload.get("animations", [])
            if isinstance(animation_evidence, dict):
                animation_evidence = animation_evidence.get("selected", [])
            selected = [str(name).casefold() for name in animation_evidence]
            if payload.get("status") != "ok" or not asset.is_file() or asset.stat().st_size < 1024:
                raise RuntimeError(f"{utc_resref}:{animation_name} export failed: {payload}")
            if animation_name.casefold() not in selected:
                raise RuntimeError(
                    f"{utc_resref}:{animation_name} missing selected-animation evidence: {payload.get('animations')}"
                )
            exports.append({
                "utc_resref": utc_resref,
                "animation": animation_name,
                "asset": str(asset),
                "bytes": asset.stat().st_size,
                "head_resref": row.get("head_resref", ""),
                "skeleton_resref": row["body_resref"],
            })
            print(f"[{completed}/{total}] {utc_resref}:{animation_name}", flush=True)
    return {
        "schema": "ghoststudio.enclave-interior-animation-sources.v1",
        "module": "m13aa",
        "character_count": len(unique),
        "animation_source_count": len(exports),
        "all_exports_valid": len(exports) == total,
        "exports": exports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=GAME_DEFAULT)
    parser.add_argument("--export-root", type=Path, default=EXPORT_DEFAULT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--animation", action="append", default=[])
    args = parser.parse_args()
    _bootstrap()
    roster_path = args.export_root / "enclave_interior_roster_v1.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))["creatures"]
    receipt = asyncio.run(_export(
        roster,
        game_root=args.game_root,
        export_root=args.export_root,
        only={str(value).casefold() for value in args.only},
        selected_animations={str(value).casefold() for value in args.animation},
    ))
    suffix = "_" + "_".join(sorted({str(value).casefold() for value in args.only})) if args.only else ""
    receipt_path = args.export_root / f"enclave_interior_animation_sources_v1{suffix}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path),
        "character_count": receipt["character_count"],
        "animation_source_count": receipt["animation_source_count"],
        "all_exports_valid": receipt["all_exports_valid"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
