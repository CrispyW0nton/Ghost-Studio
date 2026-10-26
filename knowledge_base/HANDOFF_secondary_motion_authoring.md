---
schema: ce-handoff/v1
title: Ghost Studio cloth and hair secondary-motion authoring
created_at: 2026-08-10T19:00:00Z
summary: Implement beginner-facing cloth and hair authoring in Character Builder, Custom Rigged Character, and Placeable Builder, backed by the proven ClothDynamicsK2 runtime and safe model-specific profiles.
resume_focus: Read every required source below, design the shared versioned profile contract first, then implement the workflow in phases without changing the accepted PFBNM behavior.
---

# Development handoff: universal cloth and hair authoring for Ghost Studio

You are taking over a live Windows development project spanning Ghost Studio and the KOTOR Patch Manager runtime. Read this document completely before changing anything. Preserve all uncommitted work in both repositories. Do not simplify or weaken executable fingerprint checks, hook-byte checks, memory validation, package collision checks, or backup/restore behavior.

## Mission

Make cloth and hair simulation a normal, nontechnical part of Ghost Studio's existing character and placeable workflows. A modder should be able to select a robe panel, sleeve, cape, ponytail, braid, hair lock, banner, cable, or similar secondary-motion region; choose an appropriate preset; preview and tune it; add collision shapes visually; validate it; and export a KOTOR Patch Manager-compatible package without knowing executable addresses, runtime node keys, C++, Ghidra, or the internal Part layout.

Integrate the feature into all of these existing product paths:

1. Native KOTOR Character Builder.
2. Custom Rigged Character Builder (Custom Animation Patch route).
3. Placeable Builder, including animated placeables and native Odyssey dangly meshes.

Do not create a disconnected proof-of-concept window. Build one reusable, headless **Secondary Motion** domain/workflow used by all three product surfaces. The UI must remain thin. Do not declare the feature complete when only Ghost Studio preview works: exported assets, Patch Manager packaging, runtime validation, and visible KOTOR II tests are mandatory.

## Repositories, game, and authoritative evidence

### Ghost Studio

- Repository: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio`
- Agent contract: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\AGENTS.md`
- Package ownership authority: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\knowledge_base\package_ownership_model.md`
- Knowledge routing: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\docs\knowledgebase\skills.md`
- Rigging rules: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\docs\knowledgebase\riggingskill.md`
- Character Builder principles: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\docs\knowledgebase\learned\characterbuilderprinciples.md`
- Custom-rig architecture: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\docs\architecture\custom_rigged_character_builder.md`
- Native payload generator: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\scripts\native_python_payload_generator.py`
- Ghost Studio model-validation tools: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio\native\GhostRigger.Core.Automation\Python\src\kotormcp\tools\ghostrigger_tools.py`

The Ghost Studio worktree is already dirty. Preserve it. Follow `AGENTS.md`: edit the canonical root `src\...` source first when one exists, then regenerate embedded payloads. Some newer features intentionally exist only in package-local `native\<Project>\Python\src\...`; do not invent a duplicate root module when package ownership says the package-local file is canonical.

### KOTOR Patch Manager and runtime

- Repository: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager`
- Cloth knowledge base (read completely): `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\knowledge_base\cloth_dynamics_k2.md`
- Accepted runtime source: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\Patches\ClothDynamicsK2\ClothDynamicsK2.cpp`
- Manifest: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\Patches\ClothDynamicsK2\manifest.toml`
- Hook table: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\Patches\ClothDynamicsK2\kotor2-steam-aspyr.hooks.toml`
- Address database: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\AddressDatabases\kotor2_steam_aspyr.db`
- Offline math test: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\tools\test_k2_cloth_solver_math.py`
- Capture analyzer: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\tools\analyze_k2_cloth_probe.py`
- Runtime observer starter: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\tools\start_k2_cloth_debug_session.py`
- Safe staging script: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\tools\stage_k2_player_animations.ps1`

### Debugger, Ghidra, and game

- KotorDebugger: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\KotorDebugger`
- Debugger sessions: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\KotorDebugger\sessions`
- Ghidra headless launcher: `C:\Users\NewAdmin\Documents\Ghidra\ghidra_12.1.2_PUBLIC\support\analyzeHeadless.bat`
- Ghidra project: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghidra\projects\active\kotor-2\ghidra-project`, program `kotor-2`
- Game: `C:\Program Files (x86)\Steam\steamapps\common\Knights of the Old Republic II`
- Supported live EXE SHA-256: `306F3CF9C45B8D9A086AFE10964A3512FC202477D7F8398511B297550990AE51`
- Accepted visible cloth test: `C:\Users\NewAdmin\Videos\2026-07-19 12-13-17.mp4`
- Accepted runtime session: `C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\KotorDebugger\sessions\20260719-1212-cloth-phase2c-foot-clearance`

Use the system `py` for KOTOR Patch Manager tools because that Python has PyKotor. Do not use Ghost Studio's old virtual environment for these scripts. Stage only while the game is closed and only through the approved staging script.

## Existing Ghost Studio integration points

Read these files before planning edits.

### Custom Rigged Character

- Project document: `native\GhostRigger.Core.Project\Python\src\core\project\custom_rigged_character_project.py`
- Import: `native\GhostRigger.Core.Workflow\Python\src\core\characters\custom_rigged_character_import_service.py`
- Build: `native\GhostRigger.Core.Workflow\Python\src\core\characters\custom_rigged_character_build_service.py`
- Packaging/install: `native\GhostRigger.Core.Workflow\Python\src\core\characters\custom_rigged_character_packaging_service.py`
- Validation: `native\GhostRigger.Core.Validation\Python\src\core\validation\custom_rigged_character_validator.py`
- Window: `native\GhostRigger.Core.GUI.Display\Python\src\gui\windows\qt_custom_rigged_character_builder_window.py`
- Controller: `native\GhostRigger.Core.GUI.Display\Python\src\gui\windows\qt_custom_rigged_character_builder_controller.py`
- Mode selector: `native\GhostRigger.Core.GUI.Display\Python\src\gui\windows\qt_character_builder_mode_selector.py`
- Workflow tests: `tests\test_custom_rigged_character_builder_workflow.py`

Add a stable `secondary_motion` workflow step between `materials_uvs` and `gameplay`, migrate the project schema, and preserve unknown fields and older projects.

### Native KOTOR Character Builder

- Current panel payloads: `native\GhostRigger.Core.GUI.Display\Python\src\gui\panels\qt_character_builder_panel.py` and its generated/package mirror under `native\GhostRigger.Core.Tools\Python\src\gui\panels\qt_character_builder_panel.py`
- Character principles and existing tests under `tests\test_character_builder_*.py`

Determine which file is canonical under the current payload manifest before editing. Add the same Secondary Motion editor as a tab/step after rigging and deformation proof, not a separate native-only implementation.

### Placeable Builder

- Project document: `native\GhostRigger.Core.Project\Python\src\core\project\placeable_asset.py`
- Workflow facade: `native\GhostRigger.Core.Workflow\Python\src\core\workflow\placeable_builder_service.py`
- Tool orchestration: `native\GhostRigger.Core.Tools\Python\src\core\tools\placeable_builder_tool_service.py`
- Window/controller: `native\GhostRigger.Core.Tools\Python\src\gui\windows\qt_placeable_builder.py` and `qt_placeable_builder_controller.py`
- Export integration: `native\GhostRigger.Core.Scene\Python\src\core\modules\authored_module_export.py` and its payload mirror
- Tests: `tests\test_placeable_builder_ui.py`, `tests\test_authored_module_export.py`

Add a `Secondary Motion` tab beside Visual/Particles/Resources. Preserve unknown project fields and existing stock, particle, UTP, MDL/MDX, texture, and Map Studio behavior.

## What the current runtime actually proves

The runtime is a guarded post-animation solver, not an engine replacement and not a rigging system. It hooks the end of `Gob::Animate` at `0x004E3109`, after stock animation sampling, verifies the exact original bytes `8B E5 5D C2 04 00`, and then makes bounded local-quaternion writes. The accepted hook must remain byte- and EXE-fingerprint-validated.

Proven 32-bit layout:

- `gob+0x84`: static/local Model.
- `gob+0x98`: live Part instance root. This is the simulation tree.
- Part `+0x08`: local XYZ.
- Part `+0x14`: local WXYZ quaternion.
- Part `+0x24`: scale.
- Part `+0x28`: runtime/global node key.
- Part `+0x30/+0x34`: child array/count.
- Part `+0x3C`: parent.
- Part `+0x40`: owning Gob.
- Gob world position/rotation/scale: `+0xA4`, `+0xB0`, `+0x1E0`.
- World transform composition is parent multiplied by local.

The solver maintains persistent world-space particles, uses a 120 Hz fixed step with at most six substeps, and uses a real performance counter because the game may call the animation epilogue several times per display frame. Engine `dt` is retained only as a pause/discontinuity gate. It applies gravity, damping, weak animation following, distance and cone constraints, actor-motion carry, body/limb capsules, guide planes, and moving foot guides; then it converts the result back to local joint rotation while preserving sampled twist and quaternion sign continuity.

Current accepted PFBNM values are the regression baseline, not universal magic values:

- gravity `4.25`;
- robe/sleeve damping `4.5 / 6.5`;
- robe/sleeve animation follow `0.65 / 0.45`;
- maximum speed `3.5`;
- collision velocity retention `0.20`;
- fixed step `1/120`, six maximum substeps, nine constraint iterations;
- maximum swing `135°`, robe cone `70°`, sleeve cone `115°`;
- maximum eight solver actors, three nodes per chain, nine quaternion write plans in the current build.

PFBNM currently has six hardcoded chains: front robe, left/right sleeves, and middle/left/right rear robe. The current runtime keys and rest transforms are model-specific. PMBMN has similar names at different ordinals. Therefore **names, ordinals, or guessed keys alone are never a safe write identity**.

Every frame currently captures and validates the entire writable snapshot before committing any write. Missing nodes, wrong parent/owner, a changed model/root, invalid transform, non-finite state, clock discontinuity, teleport, correction cap, duplicate destination, or model mismatch must yield zero writes and a bounded diagnostic.

### Hair and native dangly lessons

There are two valid but mutually exclusive simulation backends:

1. **ClothDynamicsK2 runtime bone-chain PBD**: use for skinned cloth/hair driver bones when collision, animation following, and controlled inertia matter.
2. **Native Odyssey dangly mesh**: use for per-vertex secondary motion when native behavior is sufficient. It does not provide reliable character-body collision.

Never apply both backends to the same vertices or driver region.

The K2 dangly mesh header is 28 bytes after the K2 mesh header. It stores constraint pointer/counts at `+0/+4/+8`, displacement/tightness/period at `+12/+16/+20`, and a rest-position pointer at `+24`. It requires one constraint and one mesh-local rest-position `vec3` per vertex. Exact-zero constraint values are **static anchors**; nonzero values are moving vertices. Ghost Studio's older writer/simulator reversed that meaning and omitted the rest-position array. Do not restore that bug. The installed PyKotor reader also has a known extra `+12` error when seeking the constraint array, so implement/use a raw binary verifier rather than trusting that reader alone.

Xaria's split body/head models (`p_xariabb`, `p_xariah6`, compatibility alias `p_xariah`) are deliberately read-only in ClothDynamicsK2 v0.2.3. The capture observes candidate hair nodes but performs `writes=0`. Keep it that way until an exact split-model profile, attachment relationship, collision space, and visible retail proof exist.

## Required product design

### One shared Secondary Motion domain

Create a reusable, Qt-free domain owned by Project/Workflow/Validation/IO according to Ghost Studio's package map. Suggested responsibilities—not necessarily filenames—are:

- Project: versioned authoring state and migrations.
- Workflow: chain discovery, driver creation, weighting assistance, collider fitting, preset application, simulation preview orchestration, profile compilation, and build handoff.
- Validation: blocking/warning rules and evidence reports.
- IO: profile serialization, raw Odyssey dangly read/write verification, and package resources.
- Rendering/GUI Helpers: viewport overlays, pinned-point/chain/collider visualization, picking, and manipulators.
- GUI Display/Tools: the same editor widget embedded into each product surface.

Do not put simulation math, profile parsing, collider fitting, or validation in a Qt window.

### Versioned authoring contract

Define and document `ghostrigger.secondary_motion_profile.v1`. Store it inside the parent `.ghostcharacter.json` or `.ghostplaceable.json`, and also export a standalone deterministic profile artifact for Patch Manager. Preserve unknown fields.

At minimum record:

- profile ID, display label, target game, schema version, backend capability version;
- exact model resref(s), MDL/MDX hashes, hierarchy fingerprint, root, and optional split attachment relationship;
- one or more regions with stable IDs and semantic types such as robe, skirt, sleeve, cape, ponytail, braid, front hair lock, banner, cable, or custom;
- backend per region: `runtime_pbd`, `native_dangly`, or `rigid`;
- mesh membership/vertex group and driver chain names;
- parent relationship and expected local rest position/quaternion/scale for every driver;
- offline preorder only as identity evidence, never as sole runtime identity;
- pin/anchor definition and skin-weight evidence;
- collider definitions by semantic node name and local shape data;
- bounded solver parameters and the preset from which they came;
- preview tests performed and validation evidence;
- runtime calibration state, capture evidence/session, compiler version, and whether the profile is write-enabled or capture-only.

Do **not** expose or persist executable addresses, raw pointers, writable memory offsets, or user-entered runtime node keys. Users choose named scene/model elements; the compiler creates guarded runtime selectors.

### Safe generic runtime registry

The present DLL hardcodes PFBNM. To make Ghost Studio exports usable, implement a bounded, merge-safe model profile registry consumed by ClothDynamicsK2. Do not generate a separate DLL per asset.

The registry/compiler must:

- bind a profile to exact model identity and a deterministic hierarchy/rest-transform fingerprint;
- compile names into expected preorder, parent relationship, rest position/quaternion/scale, chain lengths, and semantic roles;
- traverse the live Part tree read-only and obtain the runtime keys from the validated live snapshot;
- require model/root/owner/parent/rest-transform agreement before promoting a profile from capture-only to writable;
- prevalidate every destination for the entire frame before any write;
- apply hard caps from runtime capabilities—actors, chains, nodes, colliders, writes, parameter ranges, and file sizes;
- reject duplicate profile IDs, conflicting exact model identities, malformed/non-finite values, path traversal, and registry collisions;
- parse/load profiles outside the hot animation loop and never allocate or perform disk IO per frame;
- fail closed and emit bounded diagnostics;
- allow several mods to contribute independent profile files without overwriting a shared monolithic registry;
- keep the existing PFBNM profile as the immutable reference/fallback until byte-for-byte or trace-level parity is proven.

If the exact model cannot be resolved safely from offline evidence alone, automate a calibration workflow: install a **read-only capture build**, launch/observe the model, collect the bounded mapping log, validate it against the exported hierarchy, compile the writable profile, and require an explicit visible test. The user should see “Needs one safe calibration test,” not node keys or addresses.

### Placeable safety boundary

Do not assume the character hook automatically supports placeables. First create an animated placeable fixture and prove, read-only, that its live model is reached by `Gob::Animate`, that `gob+0x98` is the correct Part root, and that transforms/ownership remain valid across placement, animation, module save/load, and object destruction.

Until that proof passes:

- native dangly may be exported for suitable placeable meshes;
- runtime PBD must be visibly labeled `Needs runtime compatibility proof` and must remain capture-only;
- static placeables must never be routed through a character write profile.

After proof, support placeable presets such as banner, hanging cable, hose, flexible antenna, and fabric panel. Use placeable-local/static anchors and explicit world reset behavior. Prove lifecycle cleanup so destroyed/reloaded placeables do not leave stale solver state.

## Beginner UI/UX

Add a shared **Secondary Motion — Cloth & Hair** editor with a simple guided flow:

1. **Choose what should move**: pick a mesh region/vertex group and optional existing driver chain.
2. **Choose a preset**: Robe/Skirt, Sleeve, Cape, Ponytail/Braid, Hair Lock, Banner, Cable/Hose, Native Dangly, or Rigid.
3. **Set the attachment**: choose the fixed/root bone or placeable anchor. Offer safe automatic detection with a confidence explanation.
4. **Create or inspect drivers**: use existing bones or create a short driver chain and transfer/normalize weights. Never silently rename a foreign rig or alter unrelated weights.
5. **Fit collision**: auto-fit suggested capsules/spheres/ellipsoids/guide planes from the skeleton; allow visual adjustment with handles.
6. **Preview**: run the actual shared solver in the viewport and show pinned points, chains, colliders, gravity direction, contacts, and warnings.
7. **Test motions**: Idle, Walk, Run, Stop, Turn, Combat/raised arms, Dialogue/head turn, Teleport/reset, Low FPS, and repeated animation-sampling calls. Placeables get swing/rotation/module-reload tests.
8. **Validate and export**: explain issues in plain language and package every dependency.

Beginner users should mainly see preset strength, looseness/stiffness, damping, animation follow, collision clearance, and gravity response. Put exact numeric solver controls under Advanced. Read permitted ranges and capacities from a runtime capability descriptor rather than duplicating constants in UI code.

Hair presets should offer head/neck/upper-torso collision and optional chest/front-lock protection. Robes need pelvis/hip/thigh/shin/foot collision plus hem guide planes. Sleeves must anchor to sampled forearms and use arm collision; never anchor sleeves to the pelvis. Placeables use only proven model-local or world/static colliders.

Prevent export when the same region is assigned to both PBD and native dangly. Warn clearly that native dangly has no dependable actor-body collision.

### Viewport quality requirements

- Preview the final KOTOR hierarchy and bind/rest pose, not only the source FBX rig.
- Allow animation scrubbing without resetting the simulation on every key.
- Offer pause, single-step, reset, and 0.25×/0.5×/1× playback.
- Visualize collision penetration and constraint error.
- Show the model's forward/up axes and gravity in KOTOR coordinates.
- Preview at variable render rates while the solver remains fixed-step.
- Keep source files read-only and make driver/weight changes reversible in project state.

## Export and KOTOR Patch Manager packaging

Every exported asset must be a KOTOR Patch Manager-compatible package that includes:

- MDL/MDX and required textures/TXI;
- UTC/UTP/2DA resources as appropriate;
- the standalone secondary-motion profile;
- a manifest dependency on the ClothDynamicsK2 runtime and, for custom animation actions, the existing Custom Animation Patch;
- merge-safe profile registration rather than overwriting another mod's registry;
- hashes of all authored inputs and generated outputs;
- exact-install preview, game-closed install gate, byte-for-byte backups, and restore metadata;
- a report saying which regions are runtime PBD, native dangly, rigid, capture-only, calibrated, or write-enabled.

Do not install to Override during preview. Do not overwrite an unrelated model/profile with the same resref. Rebuild may remove only previously recorded, hash-matching owned outputs.

## Implementation phases

### Phase 0 — evidence and architecture

1. Read all files named above and inspect current dirty changes before editing.
2. Write a short architecture contract under `docs\architecture\` and add the new project schema/migration plan.
3. Identify canonical versus embedded payload files from `native\GhostRigger.PythonPayloadManifest.json`.
4. Record the owner packages and planned tests. Do not create a new native C++ project unless the package ownership model proves it is necessary.

### Phase 1 — shared project, validation, and preview core

1. Implement the versioned Secondary Motion contract and migrations.
2. Implement shared PBD preview using the accepted fixed-step order: integrate/damp velocity, predict, project length/bending, project collisions, derive velocity.
3. Implement raw native-dangly read/write/readback verification with correct anchor semantics and rest arrays.
4. Implement presets, chain discovery, driver validation, collision definitions, and deterministic reports.
5. Add focused headless tests before UI wiring.

### Phase 2 — Character Builder integration

1. Embed the shared editor in Native KOTOR Character Builder.
2. Add the `secondary_motion` step and schema migration to Custom Rigged Character.
3. Ensure custom rigs can use their own bone names; do not require PFBNM names.
4. Include profile, model identity, calibration state, and Patch Manager dependency in build/package reports.

### Phase 3 — Placeable Builder integration

1. Add the shared tab and project fields without dropping unknown fields.
2. Support native dangly for appropriate meshes.
3. Build a read-only animated-placeable probe and prove the runtime lifecycle before enabling PBD writes.
4. Carry profile resources through Placeable Builder, authored module export, Map Studio placement, and resource collision checks.

### Phase 4 — generic ClothDynamicsK2 profiles

1. Add a strict profile parser/compiler and runtime capability descriptor.
2. Move PFBNM data into the new format only behind a parity gate; retain the hardcoded reference until accepted.
3. Automate capture-only calibration and promotion to a validated profile.
4. Extend fixed capacities only if targeted stress tests and runtime memory-safety checks justify it.
5. Keep Xaria split hair capture-only until its separate proof is complete.

### Phase 5 — packaging and visible proof

1. Export/install with Patch Manager, keeping all fingerprint and hook-byte checks.
2. Test the accepted PFBNM robe/sleeve fixture for regression.
3. Test one custom character/creature hair or cape profile.
4. Test one native-dangly hair or cloth asset.
5. Test one animated placeable profile, initially capture-only and then writable only after proof.
6. Capture behind/side and motion videos; metrics alone are insufficient.

## Verification commands and operating procedure

Use targeted tests only unless the user explicitly approves a broad suite. Typical Ghost Studio checks:

```powershell
cd "C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Ghost-Studio"
python -m pytest tests\<focused_secondary_motion_test>.py -q
python scripts\native_python_payload_generator.py GhostRigger.Core.Project
python scripts\native_python_payload_generator.py GhostRigger.Core.Workflow
python scripts\native_python_payload_generator.py GhostRigger.Core.Validation
python scripts\native_python_payload_generator.py GhostRigger.Core.GUI.Display
python scripts\native_python_payload_generator.py GhostRigger.Core.Tools
python -m pytest tests\test_native_python_payloads.py -q
```

Use Ghost Studio's model tools for backend truth—model hierarchy, transforms, skinning, textures, and export comparison—but perform visible UI/workflow testing in the real Ghost Studio Debug application. Headless widgets and MCP probes do not count as visible UI proof.

Typical runtime checks:

```powershell
cd "C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager"
py tools\test_k2_cloth_solver_math.py
$env:SKIP_PAUSE = "1"
cd Patches\ClothDynamicsK2
..\create-patch.bat ClothDynamicsK2
cd ..\..
```

With the game closed, stage only through:

```powershell
.\tools\stage_k2_player_animations.ps1
```

Start the read-only observer before launch:

```powershell
py .\tools\start_k2_cloth_debug_session.py --label secondary-motion-profile-test
& "C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\bin\Release\KPatchLauncher.exe" "C:\Program Files (x86)\Steam\steamapps\common\Knights of the Old Republic II\swkotor2.exe"
```

Only one debugger can attach to the game process. Parse the newest session under `KotorDebugger\sessions` with `tools\analyze_k2_cloth_probe.py`. Use Ghidra read-only analysis when a new placeable or split-model lifecycle is not proven. Never turn an inferred offset into a write without the read-only capture and isolated-write phases.

## Required tests

Add focused tests for:

- schema migration and unknown-field preservation for character and placeable projects;
- deterministic profile round-trip and hashing;
- mutually exclusive backends per region;
- chain parent/rest-transform/fingerprint validation;
- driver weight completeness, normalization, and engine influence limits;
- collider references and bounded parameters;
- fixed-step determinism across different render rates and repeated Animate calls;
- pause, teleport, actor/model/root replacement, and stale-state cleanup;
- malformed profile, excessive counts, NaN/Inf, duplicate IDs, conflicting model identity, and path traversal all failing closed;
- native dangly exact-zero anchors, nonzero moving vertices, both arrays, pointer/count consistency, and raw binary readback;
- package dependency/profile merge behavior and collision-safe install/restore;
- PFBNM regression parity;
- Xaria remaining `writes=0` until explicitly promoted;
- animated placeables remaining capture-only until lifecycle proof passes.

## Definition of done

The work is complete only when all of the following are true:

1. A nontechnical modder can author and preview cloth or hair in both Character Builder routes without editing JSON or knowing bone-key internals.
2. A nontechnical modder can author native dangly secondary motion for a placeable, and runtime PBD for an animated placeable is either visibly proven or honestly held capture-only.
3. Ghost Studio exports a deterministic, merge-safe Patch Manager package containing the model, resources, profile, dependencies, hashes, validation, and restore data.
4. The generic runtime resolves a custom model only through exact identity plus hierarchy/rest/parent validation and fails closed on every mismatch.
5. No region is double-simulated.
6. PFBNM's accepted robe/sleeve behavior remains visibly unchanged.
7. At least one custom character/creature hair or cape asset and one placeable asset pass visible in-game tests.
8. Behind/side captures show gravity, motion follow-through, clean collision, no exploding vertices, no persistent body/foot clipping, stable stop/start behavior, and correct reset after warp/reload.
9. Project documentation, `CHANGES.md` (`Owner: LordVaderCW`), package payloads, targeted tests, and the KOTOR Patch Manager cloth knowledge base are updated with every new proven fact.

## Non-negotiable boundaries

- Do not weaken EXE hashes or hook-byte checks.
- Do not write to a new model class before read-only mapping and an isolated reversible write prove the exact target.
- Do not treat names or preorder ordinals alone as live memory identity.
- Do not expose memory addresses or runtime keys to modders.
- Do not use native dangly and runtime PBD on the same asset region.
- Do not promise native-dangly body collision.
- Do not assume placeables share character animation lifecycle.
- Do not copy source FBX, Blend, textures, or evidence videos destructively; treat them as read-only.
- Do not stage with the game open.
- Do not declare success from solver metrics or Ghost Studio preview alone; capture the visible KOTOR result.

Begin by producing a concise implementation plan tied to the owner packages and tests above, then execute it phase by phase. Keep the accepted PFBNM runtime behavior as the control group throughout the work.
