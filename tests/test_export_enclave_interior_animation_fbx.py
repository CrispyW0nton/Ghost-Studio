from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_enclave_interior_animation_fbx.py"


def test_animation_handoff_exports_one_selected_take_per_fbx():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"animation_names": [animation_name]' in source
    assert '"animation_sources"' in source
    assert "isinstance(animation_evidence, dict)" in source
    assert "animation_name.casefold() not in selected" in source


def test_animation_handoff_preserves_bas_head_and_character_texture_resolution():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'arguments["head_resref"] = row["head_resref"]' in source
    assert '"body_texture": row.get("body_texture", "")' in source
    assert '"export_rigging": True' in source


def test_animation_handoff_can_scope_a_pilot_without_changing_canonical_roster():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--only", action="append"' in source
    assert 'parser.add_argument("--animation", action="append"' in source
    assert '"enclave_interior_roster_v1.json"' in source
