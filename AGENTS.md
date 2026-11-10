# GhostRigger Agent Manual

This file is the operating contract for agents working in GhostRigger. Treat it
as project law unless the user gives a more specific instruction for the current
task.

## Prime Directives

- Work from evidence. For game-file and model-pipeline behavior, use the MCP
  validation tools before changing code.
- Keep ownership clean. Put reusable behavior in the correct core, system,
  adapter, math, IO, format, resource, or GUI layer.
- Test the thing the user actually cares about. Backend truth checks do not
  replace visible UI/workflow testing.
- Prefer focused, targeted verification. Do not run broad sweeps unless the user
  explicitly approves them for the task.
- Preserve user data. Do not silently clear scenes, overwrite source KOTOR data,
  corrupt transforms, or store large blobs in human-readable project formats.
- Record completed changes in `CHANGES.md` with owner and verification details.

## Knowledgebase And Local Books

The local book-derived working notes live under `docs/knowledgebase/`.

- Start with `docs/knowledgebase/skills.md` when a task involves mesh topology,
  vertices, extrusion/modeling, transforms, rigging/skinning, Qt UI,
  architecture, MCP validation, resource pipelines, algorithms/geometry,
  rendering/shaders, animation runtime, C++/native integration, Python
  engineering, Unreal/technical-art workflows, game/tool experience design, or
  audio/event tooling.
- Load only the topic file needed from `docs/knowledgebase/learned/`.
- Return to `docs/books/` only when the learned skill file is not detailed
  enough for the task.
- The workspace-local PDF parser dependency under `.codex_deps/` is intentionally
  kept for future PDF extraction and should remain untracked.
- The canonical package ownership model lives at
  `knowledge_base/package_ownership_model.md`. Treat it as the authority for
  GhostRigger package ownership, naming, merge decisions, and project
  boundaries.

## Current Repository Structure

GhostRigger is now a hybrid Visual Studio C++ host plus embedded Python
application. Do not treat the repository as a flat Python app.

### Canonical Source Tree

- `src/`: canonical Python source for the current application and domain logic.
  This is still the active implementation for most behavior.
- `native/`: Visual Studio C++ package tree. Each `native/GhostRigger.*`
  directory is a distinct package/project boundary with its own `.vcxproj`.
- `native/<Project>/Public/`: public C++ headers for that package.
- `native/<Project>/Private/`: private C++ implementation for that package.
- `native/<Project>/Python/src/...`: embedded Python payload copy for that
  package. These are packaged copies, not the primary edit target when a
  matching file exists under root `src/`.
- `native/<Project>/GhostRiggerPythonPayload.json`: per-project embedded Python
  manifest.
- `native/<Project>/GhostRiggerPythonPayload.rc`: per-project `RCDATA` resource
  list for the embedded Python files.
- `native/<Project>/GeneratePythonPayload.py`: project-local wrapper around
  `scripts/native_python_payload_generator.py`.
- `native/GhostRigger.PythonPayloadManifest.json`: root manifest mapping every
  non-debug native DLL project to its packaged Python payload.
- `native/templates/`: templates for new native DLL/debug-validation project
  scaffolding.

The current root payload manifest covers 18 non-debug native DLL projects and
1,118 packaged Python file references. Treat the manifest and
`tests/test_native_python_payloads.py` as the source of truth if counts drift.

### Native Package Families

Use the package namespace to understand ownership. If an existing native
project, manifest row, README, or planning note conflicts with
`knowledge_base/package_ownership_model.md`, the existing name is legacy build
state and must be migrated deliberately rather than copied into new work.

- `GhostRigger.Native.Core.Foundation`: shared native foundations, diagnostics,
  version/capability reporting, and stable handle/contract patterns.
- `GhostRigger.Native.Core.Host`: native application host executable boundary.
- `GhostRigger.Runtime.Core` and `GhostRigger.Runtime.Core.Host`: runtime C ABI,
  lifecycle, retained handles, descriptors, and diagnostics consumed by Python.
- `GhostRigger.Runtime.Shared`: shared runtime contracts/descriptors/resources
  consumed by renderer, tools, windows, and runtime packages.
- `GhostRigger.Core.Automation`: IPC, MCP, scripting bridges, external control
  APIs, command automation, automation events, and machine-facing integrations.
- `GhostRigger.Core.Unreal`: Unreal integration, Quinn skeleton mapping,
  Unreal retargeting helpers, and Unreal-specific adapter payloads.
- `GhostRigger.Core.GUI.Display`: presentation, layout, styling, signals,
  widgets, panels, dialogs, toolbars, overlays, labels, menus, visible controls,
  icons, notifications, and display-only view state.
- `GhostRigger.Core.GUI.Helpers`: interactive helper objects such as gizmos,
  dummies, manipulators, transform handles, viewport pickers, selection helpers,
  guides, snapping helpers, and drag handles.
- `GhostRigger.Core.IO`: reading, writing, importing, exporting, serialization,
  deserialization, packing, extraction, archive access, resource-file access,
  conversion, and pure format structures/contracts.
- `GhostRigger.Core.Math`: reusable transform, matrix, camera, pivot,
  projection, coordinate conversion, normal/tangent, skinning, frame, geometry,
  measurement, and viewport math.
- `GhostRigger.Core.Project`: project files, sessions, workspace state, recent
  files, settings, dirty-state policy, and save/load workflow ownership.
- `GhostRigger.Core.Qt`: Qt-facing integration boundary, including Qt IPC,
  not domain ownership.
- `GhostRigger.Core.Rendering`: renderer-neutral contracts, render state,
  materials, texture upload policy, renderer resources, backend interfaces, and
  backend implementations.
- `GhostRigger.Core.Resources`: resource discovery, identity, addresses,
  references, lifetime, cache policy, game/library lookup, and resource
  residency policy.
- `GhostRigger.Core.Scene`: scene state, objects, transforms, pivots, hierarchy,
  selection, placement, KMAX contracts, and scene serialization contracts.
- `GhostRigger.Core.Tools`: product tool orchestration such as BAS, Character
  Builder, Module Editor, Export, Pivot Controls, Resource Browser, Retargeting,
  Sequence Editor, and 2DA Browser. Tools consume lower layers; they do not own
  reusable IO, parsing, rendering, resource, math, validation, or scene rules.
- `GhostRigger.Core.Validation`: validation rules, model/resource/scene checks,
  export gates, and comparison reports.
- `GhostRigger.Core.Workflow`: reusable multi-step workflows and pipelines that
  are not just one tool and not just GUI.

Many Phase 1 native packages are diagnostic or boundary-only. Do not move real
behavior into C++ merely because a native package exists. Native migration must
prove ownership, parity, validation, and visible workflow behavior. Merge
diagnostic-only or duplicate projects when they do not justify a real runtime,
ABI, adapter, product, subsystem, dependency, or deployment boundary.

### Embedded Python Payload Rules

- Edit the canonical root `src/...` file first when a matching source file
  exists there.
- Regenerate payload copies after canonical Python changes that are packaged
  into native DLLs.
- Do not manually patch `native/<Project>/Python/src/...` copies to diverge from
  root `src/...`. `tests/test_native_python_payloads.py` checks byte identity
  for payload files whose source exists.
- If a package intentionally owns Python that has no root `src` source, document
  the owner package and why it is package-local.
- Keep each package's payload manifest, `.rc` file, `.vcxproj`, and filters in
  sync.
- Use `python scripts/native_python_payload_generator.py <Project>` to
  regenerate one package payload.
- Use `python scripts/native_python_payload_generator.py --all` only when the
  change genuinely affects many payload packages.
- Use `python scripts/native_python_payload_generator.py --write-project-generators`
  when project-local generator wrappers or build targets need repair.
- Build targets run `GeneratePythonPayload.py` before `PrepareForBuild` when the
  wrapper exists.

### Native Project Creation And Migration Rules

- Read `native/README.md`, `native/templates/README.md`,
  `knowledge_base/cpp_integration_phases.md`, and
  `knowledge_base/native_migration_plan.md` before adding or reshaping native
  packages.
- Start new native packages from `native/templates/`; do not copy an existing
  feature project and strip it down.
- Do not add parallel `.DEBUG` application projects to `GhostRigger.sln`. Debug
  validation runs through the real owning project in `Debug|x64`.
- Use canonical Visual Studio project names directly. Do not rely on solution
  folders as a substitute for package names.
- Keep ABI/package names stable unless a batch updates project files,
  references, payload manifests, tests, bridge lookups, and compatibility shims
  together.
- Do not create new projects from broad compatibility namespaces such as
  `GhostRigger.Core.GUI.Display.*`. Renderer projects already use the canonical
  `GhostRigger.Core.Rendering.*` owner; use the canonical owners in
  `knowledge_base/package_ownership_model.md` and keep old names only as
  compatibility state until a coordinated rename/merge updates all build
  surfaces.
- New native package READMEs should include owner surface, owner package, bridge
  method, C++ ownership, Python ownership, and verification expectations.
- Python package availability checks live through
  `src.adapters.native_core.package_registry` and its packaged copies.

## MCP Validation Rules

MCP tools are for backend/model-pipeline truth only. Use them for MDL loading,
vertex transforms, textures, skinning, model-pipeline comparison, and game-file
parsing.

Do not use MCP tools, headless widget construction, or backend probes as a
substitute for visual UI/workflow testing.

The MCP tools in `ghostrigger_tools.py` import GhostRigger Python modules
directly. The `PYTHONPATH` in `.cursor/mcp.json` includes the GhostRigger root.
If an MCP import fails, fix the import path. Do not ask the user to open
GhostRigger.

### Bug-Fix Order For Model/Data Pipeline Work

1. Run `compare_model_pipelines(game, resref)` to confirm the bug exists.
2. Run `inspect_mdl(game, resref)` for PyKotor ground truth.
3. Run `inspect_mdl_ghostrigger(game, resref)` for GhostRigger output.
4. Identify the divergence.
5. Fix the owning code.
6. Re-run `compare_model_pipelines(game, resref)` to confirm the fix.
7. Run only targeted regressions by default.

Never run broad/full scans unless the user explicitly approves them. Do not run
`pytest tests/test_mcp_full_scan.py` unless the user explicitly asks for the
complete 6,078-model validation.

## Testing Rules

- Prefer `python -m py_compile ...`, targeted single test files, or specific
  `pytest path::test_name` cases tied to the change.
- Do not run broad suites such as `pytest tests/`, `pytest tests/ -x`, or
  `pytest tests/ -m "not slow"` unless the user explicitly asks for them.
- For native payload edits, prefer focused cases from
  `tests/test_native_python_payloads.py`, `tests/test_native_core_package_registry.py`,
  `tests/test_native_module_package_sweep.py`, `tests/test_native_project_templates.py`,
  and any package-specific contract tests touched by the change.
- Visible behavior must be tested in the actual GhostRigger Debug application
  launched from the active Visual Studio instance.
- Visible testing is required for UI, startup, viewport, theme/layout, workflow,
  animation playback, renderer behavior, and user-facing scene operations.
- Backend probes, direct widget screenshots, and headless widget construction do
  not count as visible workflow tests.

### Default Visible Fixtures

Use the smallest fixture that proves the workflow.

- Static object workflows: `PLC_bench` for selection, selection modes, mesh
  tools, and pivot tools.
- Animation workflows: `N_DarthMalak` with the `walk` animation looped unless
  the user names another model or animation.
- Head/body coverage: Carth body with Carth head attached.
- Cloth/body coverage: Bastila body and head.
- Module, lighting, texture, material, and renderer parity workflows:
  `K2:001ebo1` / `001EBO1` unless the user names another module.

## Change Log And Commit Rules

After every completed fix or software change, update `CHANGES.md`.

Each new entry must include:

- The date.
- `Owner: LordVaderCW`.
- The relevant `T###` roadmap task ID when one applies.
- A concise summary of the change.
- The affected subsystem or files.
- Verification performed.

If the change overlaps, merges, rebases, or otherwise intersects work from
another contributor, include an `Intersects:` line naming the other user,
branch, or commit when known, plus the touched subsystem.

When adding, changing, or discovering useful embedded Python terminal commands,
update `CHEETSHEET.md` with paste-ready commands.

Commit messages must use one of these forms and include the relevant `T###` task
ID when roadmap work applies:

- `fix(scope): short description`
- `feat(scope): short description`
- `chore(cleanup): short description`
- `test(scope): short description`

The active development branch is `main`. The roadmap is
`knowledge_base/roadmap/02_roadmap_2026_05.md`. Open PRs against
`main`.

Before merging an upstream or long-lived branch:

1. Create a temporary safety branch from current `HEAD`, for example
   `codex/pre-merge-<branch>-<YYYYMMDD>`.
2. Merge and run post-merge checks.
3. Keep the merge commit and changelog entry.
4. Delete the temporary safety branch after checks pass.

## Architecture Boundaries

Before adding or moving code, identify the owning layer. Do not append new
behavior to a convenient window, panel, viewport, or helper file just because
that is where the call starts.

- `src/core/<domain>/`: headless domain models, services, validation,
  scene/project state, resource rules, import/export decisions, and workflow
  policies. Core code must not import Qt or `src.gui.*`.
- `src/systems/<system_name>/`: focused feature systems and model pipelines that
  sit above core primitives, such as BAS assembly/composition.
- `src/adapters/<technology_or_surface>/`: glue to external runtimes, renderer
  backends, Qt-facing adapters, file/runtime APIs, and integration boundaries.
- `src/gui/windows/`, `src/gui/panels/`, `src/gui/dialogs/`,
  `src/gui/viewports/`: presentation, widgets, signals, user gestures,
  theme/layout application, and calls into services.
- `src/math/`: reusable math for transforms, cameras, pivots, projection,
  coordinate systems, frame math, GPU math, and viewcube math.
- `src/io/`, `src/formats/`, `src/resources/`: file formats, resource discovery,
  serialization, deserialization, and resource lifetime concerns.
- `native/GhostRigger.*`: C++ package boundaries and embedded Python package
  copies. The namespace must mirror ownership; the existence of a native package
  does not automatically transfer behavior out of canonical Python.

The high-level ownership map is:

- IO owns file behavior: read/write/import/export, serialization,
  deserialization, MDL extraction/packing, FBX import/export, archive access,
  resource-file access, and conversion.
- Automation owns IPC, MCP, scripting bridges, external control APIs,
  background automation, command automation, events, and machine-facing
  integration.
- Tools own user-facing product workflows and orchestrate lower-level owners.
- GUI Display owns visible presentation, layout, styling, signals, widgets,
  panels, controls, icons, overlays, labels, menus, dialogs, notifications, and
  display-only view state.
- GUI Helpers own interactive helper objects such as gizmos, manipulators,
  pickers, snapping helpers, guides, dummies, and drag handles.
- Scene owns scene objects, transforms, pivots, hierarchy, selection,
  placement, KMAX contracts, and scene serialization contracts.
- Resources owns discovery, identity, addresses, references, lifetime, cache
  policy, and game/library lookup.
- Formats owns pure structure and format-level contracts; IO owns read/write.
- Math owns reusable transforms, matrices, cameras, pivots, projections,
  coordinate conversion, normals, tangents, skinning math, frame math, and
  viewport math.
- Rendering owns renderer-neutral contracts, render state, materials, texture
  upload policy, renderer resources, backend interfaces, and backend
  implementations.
- Validation owns validation rules, model/resource/scene checks, export gates,
  and comparison reports.
- Bridge packages own runtime glue; Qt packages own Qt-specific surfaces.
- Runtime/Native Core owns native ABI, lifecycle, diagnostics, retained
  handles, host services, and C/C++ bridge surfaces.
- Project/Session owns project files, user sessions, workspace state, recent
  files, settings, dirty-state policy, and save/load workflow ownership.
- Workflow/Systems owns reusable multi-step pipelines that are not merely GUI
  and not only one product tool.

### Adding New Behavior

1. Search for the existing owner with `rg` and inspect nearby packages.
2. Name the owning product surface and code package in the plan, change notes,
   or changelog.
3. If no owner exists, create a focused module/package in the correct layer and
   expose a small API for callers.
4. Keep dependency direction clean: GUI may call core/systems/adapters; adapters
   may wrap core/systems for a runtime; core and systems must stay GUI-free.
5. Let windows such as `qt_main_window.py`, `module_editor_window.py`,
   `qt_character_builder_window.py`, `qt_retarget_window.py`, and files under
   `src/gui/windows/application_core/shared/` orchestrate workflows only.
6. If a window or panel change needs more than signal wiring, widget state, or a
   short service call, extract the reusable logic to the owning subsystem.
7. Add focused tests/contracts for the owning layer. Prefer core/system tests
   without Qt, then add visible GUI checks for UI workflows.

Do not create broad new `helpers.py`, `utils.py`, or window-local function piles.

## UI And Workbench Boundaries

Respect separate product surfaces: Main Viewport/KMAX, Retarget Workbench,
Character Studio, Module Studio, Map Studio, Resource Browser, Validation,
Export, and Project/session infrastructure.

- GUI windows are composition roots and presentation shells.
- Parsing, import/export decisions, resource placement, transform algorithms,
  model composition, validation policy, renderer residency, and reusable
  workflow policy belong outside GUI files.
- Workflow-specific controls belong in their owning window or panel. For
  example, retarget mode, source/target animation choices, output animation
  naming, and retarget readiness belong in the Animation Retargeting Workbench,
  not the main viewport command bar.
- The main window may provide shared services such as `ProjectManager`, current
  scene/model access, game-library rows, file dialogs, logging, theme/layout
  registration, and command routing.
- Shared architecture such as `GhostRiggerProject`, `ResourceAddress`,
  `GameResourceProvider`, `ValidationBus`, and `ExportJob` should stay headless
  or service-oriented.
- When roadmap work begins, include a module-boundary checkpoint: owning
  studio/window, user task, source/target resource type, validation/export gates,
  and what must remain outside the slice.

## Qt, Tk, And Viewport Structure

Milestone M3 / T302 removed the legacy Tk UI. Do not reintroduce Tk modules or
root-level GUI shims.

Deleted legacy Tk files:

- `src/gui/main_window.py`
- `src/gui/character_builder_window.py`
- `src/gui/blueprint_editor.py`
- `src/gui/modular_panel.py`
- `src/gui/matrix_background.py`
- `src/gui/icon_manager.py`
- `src/gui/viewport_tk.py`
- `src/gui/viewport.py`

The last ref before deletion is commit `838831f`. The `src/gui` package root
should contain only `qt_lib.py`, `__init__.py`, and documentation. Guards live
in `tests/test_qt_only_imports.py`.

Canonical Qt import routes:

- `src.gui.rendering.frame_core.renderer`: Tk-free software frame-rendering
  backend.
- `src.gui.camera.arcball_camera`: ArcBall camera state.
- `src.gui.textures.tpc` / `src.gui.textures.txi`: TPC/TXI texture helpers.
- `src.gui.qt_lib.viewports.viewport_display`: viewport display mode state.
- `src.gui.qt_lib.viewports.viewport_navigation`: viewport navigation profiles.
- `src.gui.qt_lib.viewports.qt_viewport`: Qt viewport widgets.
- `src.gui.qt_lib.windows.qt_main_window`: Qt main window entry point.
- `src.gui.qt_lib.<category>.<module>`: canonical Qt GUI import route.

Do not add `from .viewport import ...`. That shim no longer exists.

Import `FrameRenderer`, `ArcBallCamera`, `_load_tpc_bytes`, `_is_tpc_data`,
`_clean_tex_name`, and related helpers through their canonical math/GUI routes.
Viewport display and navigation modules live under `src/gui/viewports/`; do not
add new viewport-owned modules under `src/gui/rendering/`. Keep
`src/gui/viewports/frame_renderer.py` as a thin facade.

### Viewport Module Rules

- Keep `src/gui/viewports/qt_viewport.py` as a lazy public compatibility facade.
- Keep `src/gui/viewports/viewport_core/widget.py` as a lazy widget facade.
- Shared viewport imports and helper APIs belong under
  `src/gui/viewports/viewport_core/shared/`.
- Actual viewport widgets belong under
  `src/gui/viewports/viewport_core/widgets/`.
- Patch the owning `QtViewportWidget` mixin instead of unrelated viewport
  behavior. Existing mixins include `construction`, `scene_models`,
  `display_controls`, `camera_workflow`, `measurement_controls`,
  `transform_camera`, `selection_mesh`, `history_animation`,
  `event_navigation`, `rendering_pipeline`, `overlay_layers`, `picking_hover`,
  `drag_interactions`, `resource_cache`, and `state_helpers`.
- If a new viewport feature needs many methods, create a focused mixin and add
  it to `QtViewportWidget` deliberately.
- Preserve the public import path through
  `src.gui.qt_lib.viewports.qt_viewport`.
- Update source-contract tests that assemble viewport source from split files
  when adding a viewport module those contracts need to inspect.

## Math Rules

- Shared project math helpers live under `src/math/`.
- Do not add new math helper modules under GUI, renderer backend, viewport,
  camera, or gizmo folders.
- Keep old math paths as compatibility shims only when needed.
- Import canonical math helpers directly from `src.math.*`, for example
  `src.math.frame_math`, `src.math.gpu_math`, `src.math.camera_math`,
  `src.math.transform_math`, and `src.math.viewcube_math`.
- Always name spaces in transform work: source file, bind, object, parent,
  world, camera, clip, screen, UI/gizmo, and pose space.
- Points, vectors, normals, and pivots have different transform semantics.

## Theme And Layout System

- Do not hardcode new UI colors. Add or consume tokens through
  `src/gui/libtheme/` and the active `ThemeManager`.
- Do not hardcode major GUI sizes, splitter proportions, or toolbar density.
  Add or consume layout metrics through `LayoutManager` and XML files in
  `config/themes/layouts/`.
- The Matrix look is a selectable XML theme at
  `config/themes/themes/matrix.xml`, not a global style constant.
- New GUI modules should be theme-aware and use stylesheet tokens where
  possible.
- Add `apply_ghost_theme(theme)` for custom painting.
- Add `apply_ghost_layout(layout)` for standalone windows or major dock panels
  that own splitter sizes, toolbar density, row heights, or fixed control sizes.
- New panels should have stable layout IDs so community layouts can size, hide,
  or reposition them.
- Standalone windows opened from the main shell should register with the active
  `ThemeManager` when practical, or receive the current theme/layout from their
  parent.
- New UI must be checked in Default/native, Matrix, Droid, Dark, Light, and
  Classic. Classic/Light must not ship low-contrast labels, disabled text,
  table headers, or input fields.
- Avoid blocking the UI thread during theme application. Use cached
  stylesheets/icons, debounce hot reload/settings-triggered apply calls, and do
  not trigger a second full theme apply from a `themeChanged` handler.
- When adding visible UI, update `config/themes/README.md` and
  `knowledge_base/theme_layout_system.md` if new theme tokens, layout IDs, or
  button modes are introduced.

## KMAX Scene Editor Rules

- GhostRigger is not a single-model viewer. The main viewport must be
  scene-based and usable when the scene is empty.
- Do not clear scenes silently. Destructive scene actions must be explicit, and
  dirty scenes must prompt to save.
- Double-click model import must respect the user's clear/add/cancel choice.
- `.kmax` files must be versioned, human-readable scene files.
- Do not store huge raw mesh, animation, or texture data in `.kmax` unless a
  future schema explicitly requires it.
- Preserve stable scene object IDs, source references, transforms, pivots,
  material overrides, and unknown metadata.
- Use `KMaxSceneManager` for active KMAX scene state.
- New viewport systems must support empty scenes and multi-object scenes.
- Do not replace existing viewport, gizmo, renderer, or scene systems when
  adding viewport tools.
- Pivot tools must integrate with `SceneObjectInstance.transform` and
  `SceneObjectInstance.pivot`.
- Axis/reference modes must use `TransformReferenceController`.
- New viewport controls must be theme/layout aware.
- Pivot-only edits must keep visible geometry stable.
- Preserve `.kmax` compatibility for scenes that do not yet contain pivot data.

## Module Editor And KMAP Rules

- KMAP files must be versioned and human-readable.
- Do not store heavy mesh, animation, or texture blobs in `.kmap` unless a
  future schema explicitly requires it.
- Use stable IDs for KMAP projects, modules, rooms, walkmeshes, textures,
  materials, blueprints, lights, cameras, and scene objects.
- Preserve source KOTOR module data unless the user explicitly chooses an
  export/write operation.
- New Module Editor windows and panels must be theme/layout aware and must not
  hardcode Matrix-only colors or major fixed sizes.
- Store source asset references plus editable scene overrides in KMAP.
- Preserve unknown metadata for forward compatibility.

## Resource And Renderer Rules

- Separate source references, loaded resources, decoded assets, renderer
  resources, and user-authored overrides.
- Keep resource discovery and serialization in resource/IO/format layers.
- Keep renderer residency in renderer/adapters packages.
- Track lifecycle explicitly: discover, resolve, decode, validate, cache,
  present, release/invalidate.
- For textures and materials, separate texture bytes, decoded image,
  sampler/material policy, UV mapping, lightmap handling, and backend upload.
- For imported module renderer behavior, use `K2:001ebo1` / `001EBO1` as the
  primary visible test module unless the user names another module.
- Texture, material, lightmap, and MDL parsing changes require MCP-backed
  ground truth checks.

## Rigging, Skinning, And Animation Rules

- Treat rigs as layered assets: source geometry cleanup, skeleton/bones, base
  skinning, animation controls, deformation/corrective layer, and final cleanup.
- Validate naming and side conventions before export, mirroring, or remapping.
- Check pivots and local rotation axes before diagnosing animation data.
- For skinning, compare bind pose, bone order, weights, normalized influence
  totals, and deformation at high-bend joints.
- For retargeting, compare source and target topology/skeleton assumptions
  before transferring weights or animation.
- Use `N_DarthMalak` with looped `walk` for default animation proof.
- Use visible Debug app testing for playback, pose, viewport, and workflow
  behavior.

## Mesh, Vertex, And Modeling Rules

- Treat mesh edits as topology contracts.
- Validate face winding, normals, open edges, duplicate/overlapping faces,
  isolated vertices, T-vertices, missing UVs, flipped UV faces, and degenerate
  triangles.
- Preserve stable object/subobject IDs when formats or scene contracts depend on
  them.
- For per-vertex math, name the space before transforming: object, bind, pose,
  parent, world, camera, screen, or UV.
- Transform points, vectors, normals, and tangents correctly. Normals usually
  need separate handling under non-uniform scale.
- Generated topology from extrusion, bevel, inset, or bridge tools must define
  material, UV, normal/tangent, selection, and skin-weight behavior.
- Reusable modeling algorithms belong in core/system/math owners, not windows or
  viewport mixins.

## Final Pre-Response Checklist

Before declaring work complete:

- Confirm the newest user request has been answered.
- Confirm any required `CHANGES.md` entry exists and includes
  `Owner: LordVaderCW`.
- Confirm targeted checks were run or explain why they were not.
- Confirm visible testing was used when the change touched UI/workflow behavior.
- Confirm no broad scans were run without explicit approval.
- Confirm ignored/local artifacts such as `docs/books/` and `.codex_deps/` were
  not accidentally staged.
- Confirm native embedded Python payload copies were regenerated and checked if
  canonical Python files packaged into native DLLs changed.
