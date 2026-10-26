from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "export_enclave_interior_character_glb.py"


def test_glb_export_uses_canonical_roster_and_bas_fields():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"enclave_interior_roster_v1.json"' in source
    assert 'arguments["head_resref"]' in source
    assert '"body_texture": row.get("body_texture", "")' in source
    assert '"animation_names": list(row["animation_names"])' in source


def test_glb_export_selects_lossless_interchange_profile():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"format": "glb"' in source
    assert '"export_textures": True' in source
    assert '"export_rigging": True' in source
    assert 'payload.get("bas", {})' in source


def test_glb_export_validates_every_requested_animation():
    source = SOURCE.read_text(encoding="utf-8")
    assert "set(required).issubset(selected)" in source
    assert '"all_exports_valid": len(exports) == total' in source
