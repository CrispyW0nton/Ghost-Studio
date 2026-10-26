from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_enclave_interior_characters.py"


def test_export_uses_authoritative_m13aa_git_utc_and_appearance_tables():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("danm13.rim", "danm13_s.rim", "m13aa", "read_git", "read_utc", "appearance", "heads", "baseitems"):
        assert token in source


def test_export_routes_detachable_heads_through_bas_unreal_tool():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'arguments["head_resref"] = row["head_resref"]' in source
    assert '"ghostrigger_export_model_for_unreal"' in source
    assert 'payload.get("bas", {}).get("head_resref"' in source


def test_export_preserves_outfit_texture_dialogue_and_vanilla_behavior_evidence():
    source = SCRIPT.read_text(encoding="utf-8")
    for token in ("body_texture", "armor_resref", "conversation_resref", "spawn_script", "user_defined_script", "vanilla_position"):
        assert token in source


def test_export_embeds_skeleton_family_specific_animation_sets():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "HUMANOID_ANIMATIONS" in source
    assert "PROTOCOL_DROID_ANIMATIONS" in source
    assert "ASTROMECH_ANIMATIONS" in source
    assert '"meditate"' in source
    assert '"cwalk"' in source


def test_standalone_launcher_adds_repository_root_before_importing_bootstrap_helper():
    source = SCRIPT.read_text(encoding="utf-8")
    root_insert = source.index("sys.path.insert(0, root_text)")
    helper_import = source.index("from scripts.mcp.start_kotormcp_stdio import _python_roots")
    assert root_insert < helper_import
