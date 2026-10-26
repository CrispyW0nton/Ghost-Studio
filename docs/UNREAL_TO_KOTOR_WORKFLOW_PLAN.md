# Unreal → Map Studio → KOTOR II

Product design, 2026-09-24. This specifies the next implementation, not an
already-shipped wizard. Rhen Var is the accepted first retail fixture: geometry
visible, user-confirmed walking/collision, normal process exit. Spawn placement
inside the building is deferred by the user. Save/reload is not yet measured.

## Creator experience

1. Build the level in Unreal. Choose **Send Level to Ghost Studio**. First-run
   setup locates Ghost and the Unreal project; subsequent sends remember them.
   Export the saved level and its resolved dependencies from the actual Unreal
   runtime. If unsaved edits exist, explain which revision will be exported and
   let the creator save or explicitly export a transient snapshot.
2. Ghost opens **Map Studio → Unreal Import** with a full preview and a short
   list of actionable issues. Imported buildings and props remain identifiable
   and selectable by their Unreal names. Choose or place the player spawn,
   inspect the walkable overlay, and confirm material conversion warnings.
3. Click **Build & Play in KOTOR II**. Ghost validates the snapshot, compiles the
   module, reads it back, backs up the previous test build, installs it while
   the game is closed, starts the debugger, and launches through the configured
   KPM shortcut. It loads the chosen test save and warps only where that exact
   input route has been qualified. Otherwise show the warp command and an
   explicit manual handoff. Never report input submission as a successful load.
4. Make changes in Unreal, send again, and press **Update & Play**. Reuse cached
   unchanged assets. Preserve Ghost-owned spawn, collision corrections and
   gameplay edits; show conflicts where both applications changed the same
   imported property. Retain prior working builds and offer rollback.

First-run settings: Unreal project/engine version, KOTOR II installation, KPM
shortcut and a clean test save. The creator should not need Python, archive
editing, texture resref names, coordinate math, or manual file copying.

## Architecture contract

| Owner | Responsibility |
|---|---|
| Unreal native exporter | Resolve the current scene, mesh instances, effective materials and textures; emit a versioned bundle and dependency receipt. Never parse `.umap` as the conversion engine. |
| Core.IO / Core.Math | Existing hash-bound bundle loading, safe asset references, bounds checks and coordinate/normal/winding conversion. Add exporter-version compatibility and relocatable project assets. |
| Core.Scene | Imported component identities, editable geometry, authored walkmeshes, spawn and gameplay state in KMAP. Keep source snapshot, local overrides and compiled outputs distinct. |
| Core.Workflow | Durable inspect → import → review → build → validate → stage → launch → observe jobs, revision tokens, cancellation, incremental rebuild and recovery. |
| Existing module compiler / Core.IO | Compile render surfaces into playable sectors, MDL/MDX/WOK/LYT/VIS/ARE/GIT/IFO/PTH and material resources; validate the actual packaged bytes. |
| GUI.Display / Map Studio | Import preview, issue list, spawn and walkmesh editing, progress, Build & Play and explicit result states. No compiler logic in window handlers. |
| Core.Automation | Typed, bounded MCP operations calling those same services. MCPStudio routes jobs; it does not implement a second converter. |
| Retail test adapter | Closed-game transactional staging, exact KPM route, verified debugger attachment, qualified save/warp input, capture and crash evidence. |

Existing `unreal_level_import.py` now compiles a single playable sector and
preserves component-to-node provenance. The finished editor workflow must add
an import scene model and override layer so components remain independent
authoring objects; sector assembly becomes an export projection. Preserve the
working single-sector compiler behavior while introducing this separation.
KMAP must retain bundle/source revision and component IDs across save/reopen.

Rhen Var established that one Unreal component per empty-WOK room was not a
safe default. The single-sector replacement fixed this fixture. Multi-sector
maps need explicitly validated room membership, WOK connections and visibility;
neither actor count nor material count should define room boundaries.

## Automation with understandable exceptions

- Resolve dependencies, transforms, instances within supported scope, texture
  naming and packaging automatically. Detect missing packages before export.
- Provide named conversion presets. Direct diffuse/alpha mappings already work
  for this fixture; bake supported material graphs through Unreal and report
  unsupported graphs rather than silently substituting a texture.
- Propose walkable floors and blockers, with a visible player-sized clearance
  overlay. Preserve creator corrections. Unreal navigation/collision data are
  inputs to a KOTOR WOK compiler, not interchangeable serialized navigation.
- Proposed optional Unreal markers: `GS_Spawn`, `GS_Walkable`, `GS_Blocker`,
  `GS_Ignore`. These names are a proposed convention, not current support. A
  creator without markers can make those choices in Map Studio.
- Snap spawn to a valid floor and show its facing; flag disconnected floors,
  ambiguous stacked interiors and unreachable spawn before Build & Play.
- Store per-component/property overrides and source revision. Reimport never
  silently erases Map Studio work. Present changed/deleted/conflicting items.
- Identify test session ownership. Do not replace modules during a live game
  or interrupt a different task's test. Retain build receipts and install backup.
- Surface separate results: source exported, build validated, installed, game
  launched, module visible, traversal confirmed, save/reload confirmed. A
  debugger with no crashes cannot establish geometry or gameplay correctness.

## Implementation order and release gates

| Slice | Deliverable | Required proof |
|---|---|---|
| 1. Repeatable native exporter | Unreal editor command/plugin replacing the fixed Rhen Var extraction script; project/version profile; dependency closure | Export Rhen Var and an unrelated static fixture without editing a script; transforms/material overrides and hashes agree with Unreal; source unchanged |
| 2. Map Studio import/reimport | Open bundle; independent named components; import snapshot plus overrides; spawn/walkmesh review; material diagnostics; portable KMAP | Actual Ghost Debug UI import, edit, undo/redo, save/reopen and reimport; spawn/collision edits survive an Unreal prop change |
| 3. Build & Play | Workflow service, existing compiler, transactional install, configured KPM launcher, debugger and test-save route | Fresh one-action build from Map Studio; installed hash proven; visible load, traversal, fresh-save/reload and rollback verified |
| 4. Creator-ready coverage | Installer/setup, progress/cancel/recovery, targeted format diagnostics, qualified MCP operations and cached updates | A creator completes both fixtures without shell commands or assistance; interrupted export/build/install recovers without source/save loss |

Start with static exteriors and a small interior fixture containing doorways,
multiple heights and an inside spawn. Rhen Var's exterior ground subtraction
is not sufficient for building interiors. Landscape, foliage, procedural or
Blueprint-generated geometry, dynamic gameplay and full Unreal lighting parity
need separately declared support. Unsupported content must be listed before
build. Multi-room portals require a later qualified slice, not silent guessing.

The first useful product milestone is **send a supported static Unreal level,
place the spawn in Map Studio, and Build & Play twice while preserving edits**.
This removes the repeated manual work before expanding format coverage.

## Evidence and grounding

Current implementation and limits: [UNREAL_LEVEL_IMPORT.md](UNREAL_LEVEL_IMPORT.md).
Accepted fixture: `../ZbrushMCP/knowledge/rnv-unreal-kotor2-20260924/`.
Current MOD SHA-256: `a65b6f138bd226eb38013318c286cd2f88017e06b572b41865666659de37d208`.
34 focused tests pass; reusable output equals the installed retail candidate.

Private library: Jason Gregory, Game Engine Architecture, 4th ed., PDF 483–484
(printed 466–467), source `game-design-911e9eb63f1f58149348248e562b336e`,
`F:/Documents/03_School/Academy of Art University/Books/02 - Game Development/Game Programming & Engines/Game Engine Architecture, 4th Edition - Jason Gregory.pdf`.
Direct guidance: exporter/compiler/linker separation and dependency tracking.
The UX, sector projection and overrides above are project-specific design.
Ghost `docs/knowledgebase/mapstudioskill.md` establishes Map Studio/KMAP as the
authoring surface/state and separates visual editing from export ownership.

Unreal remains source authority; Ghost owns import/edit/compile, KOTOR owns
runtime proof. MCPStudio's generic Unreal route is disabled and Ghost's spatial
route unconfigured. The private Ghost MCP live catalog was verified at 170 tools;
capture, installed resource lookup and launch readiness worked. New exporter,
workflow and MCP operations must be registered and exactly qualified before
being advertised. No generic shell or arbitrary-script operator is proposed.
