"""Export the canonical K1 m13aa interior roster as Unreal-ready BAS FBX assets."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K1_DEFAULT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
OUTPUT_DEFAULT = Path(r"C:\UE\Enclave\Saved\MCPStudio\exports\enclave_characters_v1")
HUMANOID_ANIMATIONS = (
    "walk",
    "run",
    "pause1",
    "pause2",
    "talk",
    "listen",
    "meditate",
    "sitdown",
)
PROTOCOL_DROID_ANIMATIONS = ("cwalk", "cpause1", "cpause2", "creadyr", "listen")
ASTROMECH_ANIMATIONS = ("cwalk", "cpause1")


def _bootstrap() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    return cleaned.strip("_") or "character"


def _resource_data(installation, resref, restype, module_root: str = "danm13") -> bytes:
    from pykotor.extract.installation import SearchLocation

    result = installation.resource(
        resref,
        restype,
        order=[SearchLocation.OVERRIDE, SearchLocation.MODULES, SearchLocation.CHITIN],
        module_root=module_root,
    )
    if result is None or not getattr(result, "data", None):
        raise FileNotFoundError(f"Missing {resref}.{restype.extension}")
    return bytes(result.data)


def _build_roster(game_root: Path) -> list[dict]:
    from pykotor.common.misc import EquipmentSlot
    from pykotor.extract.capsule import LazyCapsule
    from pykotor.extract.installation import Installation, SearchLocation
    from pykotor.resource.formats.twoda import read_2da
    from pykotor.resource.generics.git import read_git
    from pykotor.resource.generics.utc import read_utc
    from pykotor.resource.generics.uti import read_uti
    from pykotor.resource.type import ResourceType as RT

    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        installation = Installation(game_root)
        appearance = read_2da(
            installation.resource(
                "appearance",
                RT.TwoDA,
                order=[SearchLocation.OVERRIDE, SearchLocation.CHITIN],
            ).data
        )
        heads = read_2da(
            installation.resource(
                "heads",
                RT.TwoDA,
                order=[SearchLocation.OVERRIDE, SearchLocation.CHITIN],
            ).data
        )
        baseitems = read_2da(
            installation.resource(
                "baseitems",
                RT.TwoDA,
                order=[SearchLocation.OVERRIDE, SearchLocation.CHITIN],
            ).data
        )
        module = LazyCapsule(game_root / "modules" / "danm13.rim")
        templates = LazyCapsule(game_root / "modules" / "danm13_s.rim")
        git = read_git(module.resource("m13aa", RT.GIT))
        roster = []
        for placement_index, creature in enumerate(git.creatures):
            utc_resref = str(creature.resref)
            utc_bytes = templates.resource(utc_resref, RT.UTC)
            if not utc_bytes:
                utc_bytes = _resource_data(installation, utc_resref, RT.UTC)
            utc = read_utc(utc_bytes)
            appearance_id = int(utc.appearance_id)
            modeltype = appearance.get_cell(appearance_id, "modeltype").strip().upper()
            body_slot = "a"
            texture_variation = 1
            armor_resref = ""
            armor = utc.equipment.get(EquipmentSlot.ARMOR)
            if armor is not None:
                armor_resref = str(armor.resref)
                uti = read_uti(_resource_data(installation, armor_resref, RT.UTI))
                body_slot = baseitems.get_cell(int(uti.base_item), "bodyvar").strip().lower() or "a"
                texture_variation = int(uti.texture_variation)

            if modeltype == "B":
                body_resref = appearance.get_cell(appearance_id, f"model{body_slot}").strip()
                texture_prefix = appearance.get_cell(appearance_id, f"tex{body_slot}").strip()
                body_texture = (
                    texture_prefix + str(texture_variation).zfill(2)
                    if texture_prefix not in {"", "****"}
                    else ""
                )
            else:
                body_resref = appearance.get_cell(appearance_id, "race").strip()
                race_texture = appearance.get_cell(appearance_id, "racetex").strip()
                body_texture = race_texture if race_texture not in {"", "****"} else ""

            head_resref = ""
            normal_head = appearance.get_cell(appearance_id, "normalhead").strip()
            if normal_head not in {"", "****"}:
                try:
                    head_resref = heads.get_cell(int(float(normal_head)), "head").strip()
                except (ValueError, IndexError):
                    head_resref = ""

            stringref = int(getattr(utc.first_name, "stringref", -1))
            display_name = installation.talktable().string(stringref) if stringref >= 0 else utc_resref
            body_key = body_resref.casefold()
            if body_key == "c_drdprot":
                animation_names = PROTOCOL_DROID_ANIMATIONS
            elif body_key == "l_astro02":
                animation_names = ASTROMECH_ANIMATIONS
            else:
                animation_names = HUMANOID_ANIMATIONS
            roster.append({
                "placement_index": placement_index,
                "utc_resref": utc_resref,
                "tag": str(utc.tag),
                "display_name": display_name or utc_resref,
                "appearance_id": appearance_id,
                "modeltype": modeltype,
                "body_resref": body_resref,
                "head_resref": head_resref,
                "body_texture": body_texture,
                "armor_resref": armor_resref,
                "conversation_resref": str(utc.conversation),
                "spawn_script": str(utc.on_spawn),
                "user_defined_script": str(utc.on_user_defined),
                "vanilla_position": [
                    round(float(creature.position.x), 6),
                    round(float(creature.position.y), 6),
                    round(float(creature.position.z), 6),
                ],
                "vanilla_bearing_degrees": round(math.degrees(float(creature.bearing)), 6),
                "animation_names": list(animation_names),
            })
    return roster


async def _export_roster(roster: list[dict], game_root: Path, output_root: Path) -> dict:
    from kotormcp.tools import handle_tool

    unique: dict[str, dict] = {}
    for row in roster:
        unique.setdefault(row["utc_resref"].casefold(), row)
    exports = []
    for index, row in enumerate(unique.values(), start=1):
        utc_resref = row["utc_resref"]
        asset_name = f"SK_{_safe_name(utc_resref)}"
        destination = output_root / _safe_name(utc_resref)
        arguments = {
            "game": "k1",
            "game_path": str(game_root),
            "resref": row["body_resref"],
            "body_texture": row["body_texture"],
            "export_dir": str(destination),
            "output_name": asset_name,
            "export_rigging": True,
            "export_textures": True,
            "animation_names": row["animation_names"],
        }
        if row["head_resref"]:
            arguments["head_resref"] = row["head_resref"]
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            response = await handle_tool("ghostrigger_export_model_for_unreal", arguments)
        payload = json.loads(response["text"])
        if payload.get("status") != "ok":
            raise RuntimeError(f"{utc_resref} export failed: {payload}")
        asset = Path(payload["asset"])
        if not asset.is_file() or asset.stat().st_size < 1024:
            raise RuntimeError(f"{utc_resref} did not produce a valid FBX")
        if row["head_resref"] and payload.get("bas", {}).get("head_resref", "").casefold() != row["head_resref"].casefold():
            raise RuntimeError(f"{utc_resref} BAS head evidence is missing")
        exports.append({
            "utc_resref": utc_resref,
            "asset": str(asset),
            "bytes": asset.stat().st_size,
            "body_resref": row["body_resref"],
            "head_resref": row["head_resref"],
            "body_texture": row["body_texture"],
            "animations": payload.get("animations", []),
            "texture_sidecars": payload.get("textures", {}).get("sidecars", []),
            "bas": payload.get("bas", {}),
        })
        print(f"[{index}/{len(unique)}] {utc_resref}: {asset.name}", flush=True)
    return {
        "schema": "ghoststudio.enclave-interior-character-export.v1",
        "module": "m13aa",
        "game": "K1",
        "placement_count": len(roster),
        "unique_character_asset_count": len(exports),
        "all_exports_valid": len(exports) == len(unique),
        "exports": exports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=K1_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    _bootstrap()
    args.output_root.mkdir(parents=True, exist_ok=True)
    roster = _build_roster(args.game_root)
    (args.output_root / "enclave_interior_roster_v1.json").write_text(
        json.dumps({"schema": "ghoststudio.enclave-interior-roster.v1", "module": "m13aa", "creatures": roster}, indent=2),
        encoding="utf-8",
    )
    receipt = asyncio.run(_export_roster(roster, args.game_root, args.output_root))
    (args.output_root / "enclave_interior_export_receipt_v1.json").write_text(
        json.dumps(receipt, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "placement_count": receipt["placement_count"],
        "unique_character_asset_count": receipt["unique_character_asset_count"],
        "all_exports_valid": receipt["all_exports_valid"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
