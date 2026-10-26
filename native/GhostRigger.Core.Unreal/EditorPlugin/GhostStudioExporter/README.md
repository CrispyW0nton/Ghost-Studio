# Ghost Studio native Unreal exporter

Owner surface: Unreal Editor → Tools → Export Level to Ghost Studio.
Owner package: GhostRigger.Core.Unreal. Version 0.1.0 qualifies only
UE **5.8.2-56702186+++UE5+Release-5.8**. Other versions fail explicitly.

This content plugin runs Python through Unreal's native editor APIs. It does
not run inside Ghost's embedded interpreter and is not a DLL payload copy.
There is no network, AI, MCP, or agent dependency during export.

Copy this plugin folder into a project's `Plugins` directory and enable
**GhostStudioExporter** in the editor. Save the level and referenced assets,
then use the Tools action. It writes a new snapshot under the project's Saved
directory and displays its path. Automatic Ghost launch/send is not implemented.

In Ghost, save an empty KMAP and choose **File → Import / Reimport Unreal
Level**. Select `snapshot.json`. The static conversion dialog requires explicit
component inclusion, a collision component/section, spawn, and diffuse-only
acceptance. Complex materials require a reviewed, source-revision-bound policy
with baked texture sidecars. Unsupported inputs report the source object and
repair; no shader or collision fallback silently substitutes missing data.

The same exporter can run in UnrealEditor-Cmd with the PythonScript commandlet:

```text
-run=pythonscript -script="<plugin>/Content/Python/ghoststudio_unreal_exporter.py" -GhostStudioMap=/Game/YourMap -GhostStudioOutput="<new snapshot folder>" -unattended -NullRHI
```

The commandlet opens the native map, reads native meshes/material slots and
dependency packages, and verifies that source files did not change. It never
saves the map. Inspection can report unsupported content; a snapshot is not a
successful game conversion by itself.

Verification: identical repeat native exports of the preserved Rhen Var map;
an unrelated saved floor/mirrored-prop native fixture; conversion/compiler
readback and focused IO/state tests. The editor menu and Ghost's actual Debug
UI have not yet been visually qualified. Build & Play, KPM/debugger launch,
retail save/reload, and typed MCP jobs remain separate unfinished work.
