"""Export KOTOR 1's authored PC HUD layouts and visual resources for Unreal.

The export is a clean reconstruction input, not executable game code.  GUI
geometry comes from retail ``.gui`` GFF resources and image data comes only
from the base installation's TexturePacks/BIF archives.  Loose Override and
module resources are deliberately excluded so the receipt can prove which
retail data drove the Enclave HUD.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K1_DEFAULT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
OUTPUT_DEFAULT = Path(r"C:\UE\Enclave\Saved\MCPStudio\imports\kotor_hud_k1_v1")

LAYOUTS = ("mipc8x6", "dialog", "maininterface", "container", "pause")
DYNAMIC_TEXTURES = (
    "i_dialog",
    "i_attack",
    "i_opendoor",
    "i_openplace",
    "i_useplace",
    "i_useitem",
    "i_examine",
    "i_noaction",
    "friendlyreticle2",
    "hostilereticle2",
    "friendlyarrow",
    "hostilearrow",
)


def _bootstrap() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _retail_texture(installation: object, resref: str) -> tuple[bytes, str, int] | None:
    from core.assets.resource_manager import RES_TGA, RES_TPC, _key

    for resource_type in (RES_TPC, RES_TGA):
        for archive in installation._tex_erfs:  # type: ignore[attr-defined]
            raw = archive.read(resref, resource_type)
            if raw:
                return bytes(raw), str(archive.path), int(resource_type)
        slot = installation._key_map.get(_key(resref, resource_type))  # type: ignore[attr-defined]
        if slot is not None:
            bif_index, _ = slot
            bif = installation._bif_index.get(bif_index)  # type: ignore[attr-defined]
            raw = installation.get_bif(resref, resource_type)  # type: ignore[attr-defined]
            if raw and bif is not None:
                return bytes(raw), str(bif.path), int(resource_type)
    return None


def export_hud(game_root: Path, output_root: Path) -> dict[str, object]:
    _bootstrap()
    from core.assets.resource_manager import _GameInstall
    from PIL import Image
    from src.core.rendering.kotor_gui_preview import decode_kotor_gui_texture
    from src.core.tools.kotor_gui_document import KotorGuiDocument

    if not (game_root / "chitin.key").is_file():
        raise FileNotFoundError(f"KOTOR chitin.key not found under {game_root}")
    installation = _GameInstall(str(game_root), "K1")
    layout_root = output_root / "layouts"
    texture_root = output_root / "textures"
    layout_root.mkdir(parents=True, exist_ok=True)
    texture_root.mkdir(parents=True, exist_ok=True)

    layout_rows: list[dict[str, object]] = []
    texture_names: set[str] = set(DYNAMIC_TEXTURES)
    for resref in LAYOUTS:
        raw = installation.get_bif(resref, 2047)
        if not raw:
            raise FileNotFoundError(f"Missing base-game K1 GUI resource: {resref}.gui")
        # A third-party PyKotor build used by one native payload emits verbose
        # acquire diagnostics.  They are not export evidence, so keep receipts
        # deterministic and quiet while compiling the immutable snapshot.
        with contextlib.redirect_stdout(io.StringIO()):
            document = KotorGuiDocument.from_bytes(
                bytes(raw), game="K1", resref=resref, source_kind="K1 KEY/BIF base game"
            )
            snapshot = document.preview_snapshot()
        payload = snapshot.to_pie_payload()
        target = layout_root / f"{resref}.json"
        encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        target.write_bytes(encoded)
        for control in snapshot.controls:
            texture_names.update(control.texture_resrefs)
        slot = installation._key_map.get(f"{resref}:2047")
        source_archive = ""
        if slot is not None:
            bif = installation._bif_index.get(slot[0])
            source_archive = str(getattr(bif, "path", ""))
        layout_rows.append(
            {
                "resref": resref,
                "source_archive": source_archive,
                "source_kind": "K1 KEY/BIF base game",
                "source_sha256": _sha256(bytes(raw)),
                "output_json": str(target),
                "output_sha256": _sha256(encoded),
                "canvas": [snapshot.source_width, snapshot.source_height],
                "control_count": len(snapshot.controls),
            }
        )

    texture_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for resref in sorted(name for name in texture_names if name):
        source = _retail_texture(installation, resref)
        if source is None:
            # Font resources are declared alongside image resources in GUI
            # controls and are rendered as Unreal text, not decoded as images.
            if resref.startswith(("fnt_", "dialogfont")):
                continue
            missing.append(resref)
            continue
        raw, archive, resource_type = source
        decoded = decode_kotor_gui_texture(raw, max_size=2048)
        image = Image.frombytes("RGBA", (decoded.width, decoded.height), decoded.rgba)
        target = texture_root / f"{resref}.png"
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = buffer.getvalue()
        target.write_bytes(encoded)
        texture_rows.append(
            {
                "resref": resref,
                "source_archive": archive,
                "source_kind": "K1 TexturePacks/BIF base game",
                "source_resource_type": resource_type,
                "source_sha256": _sha256(raw),
                "output_png": str(target),
                "output_sha256": _sha256(encoded),
                "dimensions": [decoded.width, decoded.height],
                "txi": decoded.txi,
            }
        )

    required_dynamic = set(DYNAMIC_TEXTURES)
    exported = {str(row["resref"]) for row in texture_rows}
    receipt: dict[str, object] = {
        "schema": "ghoststudio.enclave.kotor1-hud-export.v1",
        "game": "K1",
        "game_root": str(game_root),
        "output_root": str(output_root),
        "evidence_separation": {
            "retail_gui_data": "authored layout/control/texture references",
            "retail_texture_data": "authored visual resources",
            "unreal_runtime": "clean reconstruction; no executable code copied",
        },
        "layouts": layout_rows,
        "textures": texture_rows,
        "missing_non_font_textures": missing,
        "validation": {
            "all_layouts_exported": len(layout_rows) == len(LAYOUTS),
            "mipc8x6_is_800x600": next(
                (row["canvas"] == [800, 600] for row in layout_rows if row["resref"] == "mipc8x6"),
                False,
            ),
            "dialog_controls_present": next(
                (int(row["control_count"]) >= 3 for row in layout_rows if row["resref"] == "dialog"),
                False,
            ),
            "all_dynamic_textures_exported": required_dynamic.issubset(exported),
            "no_override_or_module_sources": all(
                "override" not in str(row["source_archive"]).lower()
                and "modules" not in str(row["source_archive"]).lower()
                for row in (*layout_rows, *texture_rows)
            ),
        },
    }
    receipt_path = output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if missing or not all(bool(value) for value in receipt["validation"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"KOTOR HUD export validation failed: {receipt_path}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=K1_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    result = export_hud(args.game_root.resolve(), args.output_root.resolve())
    print(json.dumps(result["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
