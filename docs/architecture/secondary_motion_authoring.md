# Secondary Motion Authoring Architecture

Status: implementation contract for `ghostrigger.secondary_motion_profile.v1`

## Safety baseline

The accepted KOTOR II `PFBNM` robe and sleeve solver remains the immutable
runtime regression control. Its hard-coded writer is retained until a separate
parity review accepts a compiled replacement. Every other model profile starts
`capture_only`; authoring, preview, native-dangly export, and read-only runtime
calibration do not imply permission to write live transforms.

Runtime PBD write promotion is model-specific. A profile may become writable
only after exact model hashes, hierarchy fingerprint, root/owner/parent links,
local rest transforms, bounded live-tree capture, lifecycle behavior, and a
visible game test all pass. Xaria split hair remains capture-only. Placeables
also require proof that `Gob::Animate` reaches the intended animated instance,
that `gob+0x98` is its live `Part` root, and that placement, save/load,
animation, destruction, and reload cleanly reset solver state. Static
placeables never use a character writer.

No profile stores executable addresses, pointers, memory offsets, or runtime
node keys. Users select model elements by name; the compiler produces guarded,
bounded selectors from exact identity evidence.

## Ownership

| Responsibility | Owning package | Contract |
| --- | --- | --- |
| Versioned profile state and parent-project migrations | `GhostRigger.Core.Project` | Preserve unknown fields; embed one profile in `.ghostcharacter.json` or `.ghostplaceable.json`. |
| Deterministic standalone serialization and raw Odyssey dangly verification | `GhostRigger.Core.IO` | Stable JSON bytes/hashes; verify both constraint floats and local rest-position arrays; zero constraint means fixed anchor. |
| Presets, chain discovery, driver checks, collider fitting, fixed-step preview, compiler, build handoff | `GhostRigger.Core.Workflow` | Qt-free orchestration; custom bone names are valid; runtime capacities come from a descriptor. |
| Blocking/warning rules and promotion evidence | `GhostRigger.Core.Validation` | Fail closed; reject mixed backends, ambiguous identity, unbounded values, and unproven write requests. |
| Overlays and manipulators | `GhostRigger.Core.Rendering` / `GhostRigger.Core.GUI.Helpers` | Presentation data, picking, and handles only; no simulation policy. |
| Shared beginner editor and host-window wiring | `GhostRigger.Core.GUI.Display` / `GhostRigger.Core.Tools` | One editor contract embedded in Custom, Native KOTOR, and Placeable builders. |
| Runtime profile registry and guarded live capture | KOTOR Patch Manager `ClothDynamicsK2` | Parse outside the hot loop, enforce hard caps, support merge-safe independent files, and keep PFBNM as fallback. |

Package-local Python files remain canonical where the native payload manifest
has no matching root `src/` source. Mirrored payloads must stay byte-identical.
No new native Visual Studio project is required.

## Profile contract

`ghostrigger.secondary_motion_profile.v1` records:

- stable profile identity, label, target game, schema and capability versions;
- exact model resrefs and MDL/MDX hashes, hierarchy fingerprint, root, and
  optional split-attachment relationship;
- stable regions with semantic type, exactly one of `runtime_pbd`,
  `native_dangly`, or `rigid`, mesh/vertex membership, named drivers, parent
  names, local rest position/quaternion/scale, and offline preorder evidence;
- anchor and skin-weight evidence, semantic-node colliders with local shape
  data, bounded solver values, and preset provenance;
- preview tests and deterministic validation evidence;
- runtime calibration/capture evidence, compiler version, and the explicit
  `capture_only` or `write_enabled` state.

Unknown profile, region, driver, collider, evidence, and parent-project fields
round-trip unchanged. Standalone profile files use sorted UTF-8 JSON with LF
termination so Patch Manager can hash and register them deterministically.

## Backend and export gates

Each region has exactly one backend. A mesh region cannot be both runtime PBD
and native dangly. `rigid` requires no runtime dependency. `native_dangly` can
be written to a model only after the raw MDL readback proves one finite
constraint and one finite local rest position per rendered vertex. Runtime PBD
always carries the `ClothDynamicsK2` dependency, including capture-only
profiles.

Custom characters, native character hair, and placeables may be packaged with
capture-only profiles. Packaging reports must clearly separate previewed,
native-dangly, rigid, capture-only, calibrated, and write-enabled regions.
Only a model-specific accepted validation report can request runtime writes.

## Fixed-step preview

The shared preview uses a fixed 120 Hz step with at most six catch-up substeps:
integrate and damp velocity, predict positions, project length and bending
constraints, project collisions, then derive velocity. Rendering rate is
independent. Pause, reset, single-step, 0.25x/0.5x/1x playback, and repeated
animation-sampling calls must not silently reset state.

## Phased verification

1. Project/IO/Workflow/Validation: migration and unknown-field round trips,
   deterministic bytes, bounded validation, fixed-step determinism, collision
   and reset behavior, and raw native-dangly readback.
2. Character builders: shared step/tab presence, project synchronization,
   custom driver names, package profile/dependency/report output, and no
   non-PFBNM write promotion.
3. Placeable builder/export: shared tab, profile/resource collision handling,
   package carry-through, and a capture-only lifecycle report.
4. Runtime: malformed/collision/capacity rejection, load-time parsing, guarded
   read-only capture, and source-contract proof that PFBNM is the sole writer.
5. Visible proof: accepted PFBNM robe/sleeve regression, one custom hair/cape,
   one native-dangly asset, and one animated placeable capture. Metrics support
   but never replace behind/side motion video.
