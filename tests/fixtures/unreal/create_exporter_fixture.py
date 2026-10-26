"""Run only in the disposable GhostExporterFixture Unreal project."""
from pathlib import Path
import unreal

project = Path(unreal.Paths.get_project_file_path()).resolve()
if project.stem != "GhostExporterFixture" or (project.parent / "Content/StaticFixture.umap").exists():
    raise RuntimeError("Fixture authoring requires a fresh, explicitly named disposable project.")
world = unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
tools = unreal.AssetToolsHelpers.get_asset_tools()
material = tools.create_asset("FixtureMaterial", "/Game", unreal.Material, unreal.MaterialFactoryNew())
node = unreal.MaterialEditingLibrary.create_material_expression(material, unreal.MaterialExpressionConstant3Vector)
node.set_editor_property("constant", unreal.LinearColor(0.25, 0.5, 0.75, 1))
unreal.MaterialEditingLibrary.connect_material_property(node, "", unreal.MaterialProperty.MP_BASE_COLOR)
unreal.MaterialEditingLibrary.recompile_material(material)
unreal.EditorAssetLibrary.save_loaded_asset(material)
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for label, mesh_path, location, scale in (
    ("WalkableFloor", "/Engine/BasicShapes/Plane.Plane", (0, 0, 0), (10, 10, 1)),
    ("MirroredProp", "/Engine/BasicShapes/Cube.Cube", (200, 200, 100), (-1, 2, 1)),
):
    actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(*location))
    actor.set_actor_label(label)
    actor.static_mesh_component.set_static_mesh(unreal.load_asset(mesh_path))
    actor.static_mesh_component.set_material(0, material)
    actor.set_actor_scale3d(unreal.Vector(*scale))
if not unreal.EditorLoadingAndSavingUtils.save_map(world, "/Game/StaticFixture"):
    raise RuntimeError("Could not save isolated native fixture.")
unreal.log("GHOST_STUDIO_FIXTURE_CREATED")
