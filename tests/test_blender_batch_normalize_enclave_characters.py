from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "blender_batch_normalize_enclave_characters.py"


def test_batch_normalizer_uses_canonical_glb_and_roster_receipts():
    source = SOURCE.read_text(encoding="utf-8")
    assert '"enclave_interior_roster_v1.json"' in source
    assert '"enclave_interior_character_glb_v1.json"' in source
    assert '"--roster"' in source
    assert "args.roster" in source
    assert "if args.roster is not None" in source
    assert "required_by_utc" in source


def test_batch_normalizer_creates_mesh_only_base_and_single_action_fbxs():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _export_base(" in source
    assert "bake_anim=False" in source
    assert "def _export_action(" in source
    assert "bake_anim_use_all_actions=False" in source
    assert "use_armature_deform_only=True" in source
    assert 'object_types={"ARMATURE"}' in source
    assert 'armature.select_set(True)' in source
    assert "bpy.context.scene.render.fps = 30" in source


def test_batch_normalizer_requires_one_armature_meshes_and_all_actions():
    source = SOURCE.read_text(encoding="utf-8")
    assert "len(armatures) != 1 or not meshes or missing" in source
    assert '"all_characters_valid": len(records) == len(glb_exports)' in source


def test_batch_normalizer_aligns_actual_open_neck_loops_in_three_dimensions():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _head_neck_boundary_correction(" in source
    assert "_boundary_components(" in source
    assert "seam_correction = Matrix.Translation(tuple(seam_correction_vector))" in source
    assert '"neck_seam_correction_m"' in source


def test_batch_normalizer_preserves_authored_facial_frames_when_building_rest_skin():
    source = SOURCE.read_text(encoding="utf-8")
    assert "authored_attachment_world =" in source
    assert "def _preserve_authored_attachment_frames(" in source
    assert "head_frame_delta = rest_world[head] @ authored_world[head].inverted_safe()" in source
    assert "target_world = head_frame_delta @ authored_world[mesh]" in source
    assert "facial_offset_corrections = _preserve_authored_attachment_frames(" in source
    assert '"facial_offset_correction_m"' in source


def test_batch_normalizer_purges_fake_user_actions_between_characters():
    source = SOURCE.read_text(encoding="utf-8")
    assert "for action in list(bpy.data.actions):" in source
    assert "bpy.data.actions.remove(action)" in source


def test_batch_normalizer_strips_malformed_tongue_faces_after_armature_import():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _strip_malformed_bas_tongue_faces(" in source
    assert 'obj.name.casefold() == "tongue"' in source
    assert 'context="FACES_ONLY"' in source
    import_marker = "bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True)"
    bind_marker = "_bake_head_attachment_into_armature("
    cleanup_marker = "stripped_malformed_meshes = _strip_malformed_bas_tongue_faces()"
    export_marker = "_export_base(base_path)"
    import_index = source.index(import_marker)
    bind_index = source.index(bind_marker, import_index)
    assert import_index < bind_index
    assert bind_index < source.index(cleanup_marker)
    assert source.index(cleanup_marker) < source.index(export_marker)
    assert '"stripped_malformed_meshes": stripped_malformed_meshes' in source


def test_batch_normalizer_converts_uv_origin_once_at_fbx_boundary():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _convert_uv_origin_for_fbx(" in source
    assert "loop.uv.y = 1.0 - float(loop.uv.y)" in source
    conversion = "uv_origin_converted_meshes = _convert_uv_origin_for_fbx()"
    export = "_export_base(base_path)"
    assert source.index(conversion) < source.index(export)
    assert '"uv_origin_converted_meshes": uv_origin_converted_meshes' in source


def test_batch_normalizer_materializes_each_base_color_family_before_fbx_export():
    source = SOURCE.read_text(encoding="utf-8")
    assert "def _materialize_base_color_images(" in source
    assert "def _upstream_image_node(" in source
    assert "_upstream_image_node(input_socket, visited)" in source
    assert "materialized_texture_files = _materialize_base_color_images(" in source
    assert source.index("materialized_texture_files = _materialize_base_color_images(") < source.index("_export_base(base_path)")
    assert "_material_family(material.name)" in source
    assert "object-space rig conversion must never rewrite UV or sampler identity" in source
