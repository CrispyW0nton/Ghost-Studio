# Export a whole level

In the **Map Studio / Level Editor** window, open the KMAP and choose:

- **File → Export Whole Level as FBX…**
- **File → Export Whole Level as OBJ…**

These commands export the current enabled, visible room geometry together in
one file, keeping individual meshes separately named. Stock room instances and
authored room geometry are supported. Authored creatures, doors and placeables
contribute their resolved visual models in a static rest pose. Editor markers,
player preview figures and collision-only meshes are excluded.

Room and module placement is baked into the mesh coordinates once. Do not apply
the original LYT room offsets again after import. Check the receiving program's
axis and unit settings: the exported coordinates use KOTOR's Z-up meter space;
FBX declares its units, while OBJ has no standard unit declaration.

The export includes diffuse material references and copies available textures
to a folder beside the mesh. Keep these sidecars together when moving files.
FBX and OBJ use different texture folders so exporting both formats with the
same filename stem is safe. OBJ also writes an MTL file.

FBX preserves the second UV set for lightmaps. Lightmap images and their mesh
associations are included as sidecars/manifest metadata; this does not recreate
KOTOR's lighting shader automatically. OBJ supports only the primary UV set.
Animations, scripts, triggers, pathfinding and playable module behavior are not
part of either mesh format. The JSON manifest retains scene/gameplay metadata;
the separate **Export .mod Package…** workflow creates a KOTOR game module.

Missing required room or placement models stop export with an error. Missing
textures are reported and their resource references are retained. Geometry
export does not require a game-ready walkmesh. A failed export leaves previous
output files intact.

## Main-window commands

The main scene window and the Level Editor have separate File menus. Current
source on the default `ghost-studio` branch provides additional commands.
Select a room in the loaded module layout in the main window
and use **File → Export Full Module as One FBX…** or **File → Export Clean Full
Map as OBJ…**. The clean-map OBJ workflow filters backdrop geometry. These are
different commands from the Level Editor's whole-level export. Older builds from
`4d15cfc5` have neither command and have the manifest-only Level Editor
implementation. Build the updated source to obtain these features; updating
GitHub source does not replace an already downloaded executable.

## AI and MCP access

An AI can only call the tools connected to its current session. Access to
GModular or Unreal does not by itself establish access to Ghost Studio.

Ghost Studio's Automation package includes several distinct integration paths.
The narrow `ghoststudio_spatial_mcp` connector exposes health, spatial snapshots,
captures and evidence-gap reporting. It does not expose arbitrary mouse/keyboard
control or a level-export command. The broader KotorMCP integration provides
game-resource/model tools; a model-export tool is not proof that an AI can
operate the current Level Editor scene.

The broad KotorMCP server now exposes **`ghoststudio_export_level`**. Start it
with `python scripts/mcp/start_kotormcp_stdio.py` from this checkout and reconnect
the client so it refreshes its tool catalog. Save the level as a KMAP first.
For example, call the tool with paths on the machine running the server:

```json
{
  "kmap_path": "C:/Projects/end_m01aa.kmap",
  "output_path": "C:/Exports/end_m01aa.obj",
  "game_dir": "C:/Games/Knights of the Old Republic"
}
```

Use an `.fbx` output path for FBX. Supply `asset_dir` if custom MDL/MDX,
templates or textures are needed. The tool exports the saved scene, including
its room placement, without opening or controlling the desktop window. It
does not include unsaved edits. It reports file size, SHA-256 and geometry counts,
and uses MCP error status on failure. Existing mesh and sidecar files require
explicit `overwrite: true`; all output paths must resolve outside the game
installation. The spatial connector's four-tool catalog remains unchanged.

When diagnosing an AI-assisted workflow, record the Ghost Studio version, the
window being used, the connected MCP server names, and the actual available
tool names. Do not infer menu paths or control capabilities from the product
name. See `native/GhostRigger.Core.Automation/README.MCP.md` for the connector
boundaries.
