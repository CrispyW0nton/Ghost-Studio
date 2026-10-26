# Ghost Studio program audit — 2026-09-24

Owner: LordVaderCW. Reviewed the dirty working tree based on `e1755571`;
existing contributor changes were preserved. This is a risk-directed source and
backend audit, not an exhaustive whole-program or native UI certification.

Eight actionable findings are retained below. P1 means prioritize before relying
on the affected workflow; P2 means a reproducible correctness or packaging defect
with narrower impact. No production fixes were applied during the audit.

## Findings

1. **P1 — Multi-directory export rollback corrupts same-named originals.**
   `native/GhostRigger.Core.IO/Python/src/core/export/export_job.py:217`
   names every backup only with `final.name`. Exporting `a/same.bin` and
   `b/same.bin` overwrites the first backup with the second. Injecting failure on
   a third promotion restored `original-1` into the first destination and left
   the second missing. Both original files should survive. Allocate unique
   backup paths by output identity, and test rollback across directories.

2. **P1 — A failed restore deletes the last recoverable backup.**
   The same file's unconditional cleanup at lines 264 and 301 removes staging
   and backup directories even when `_rollback_promoted_outputs` reports a
   restore failure. A destination lock after backup creation reproduces a failed
   job with no original and no retained backup. The clean OBJ caller also
   unconditionally removes its stage directory in
   `native/GhostRigger.Core.GUI.Display/Python/src/gui/windows/application_core/shared/model_io.py:357`.
   Keep unresolved backups in durable recovery storage and return their paths;
   caller cleanup must honor that state.

3. **P1 — Interrupted scene saving destroys the previous save.**
   `native/GhostRigger.Core.Scene/Python/src/core/scene/kmax_serializer.py:25`
   writes directly over the destination. A simulated disk-full error left a
   566-byte valid save as 22 bytes of invalid JSON. The project writer has the
   same direct-write pattern at
   `native/GhostRigger.Core.Project/Python/src/core/project/ghostrigger_project.py:439`
   (source-confirmed; the injected runtime probe covers KMAX). Write and verify a
   sibling temporary file before replacement. Commit `scene.path` and clean
   state only after success; Save As currently changes the path before IO.

4. **P1 — DLL discovery order selects incompatible Python implementations.**
   `native/GhostRigger.Native.Core.Host/main.py:98` enumerates DLLs without an
   owner policy, and line 118 silently accepts the first module definition.
   Running the actual registration method with the IO/Tools manifests in either
   order selects different `src.core.export.export_job` implementations. The
   Tools copy lacks `promote_staged_files`, imported by Clean Module OBJ at
   `model_io.py:313`, and also lacks IO's protection against files appearing
   after no-overwrite preflight. There are 20 distinct duplicate module paths
   with different byte hashes (some are initializers and need semantic triage).
   Establish one authoritative module owner, remove obsolete duplicates or
   reject mismatched definitions. Merely sorting DLLs does not resolve ownership.
   This is importer-level proof, not a claim that a particular running DLL build
   has already hit the failure.

5. **P2 — Regular model loading can combine two resource revisions.**
   `native/GhostRigger.Core.Resources/Python/src/core/assets/resource_manager.py:711`
   reads MDL and MDX separately without the lock used by `load_model_strict`.
   Publishing a second project overlay between those reads passes `old-mdl`
   and `new-mdx` to the parser. Background resource changes can therefore cause
   parse failures or corrupted geometry. Capture both resources and provenance
   under one revision, then parse outside the lock. The existing concurrency
   test covers the strict method only.

6. **P2 — Head skin transfer differs between source and embedded payload.**
   `src/math/head_skin_transfer.py:164` accepts up to 32 slots, but
   `native/GhostRigger.Core.Math/Python/src/math/head_skin_transfer.py:162`
   still rejects more than 16. The source regression for a 17-slot palette is
   therefore not sufficient evidence for the embedded application. Synchronize
   this owner deliberately alongside its known manifest conflict; do not
   regenerate all packages over unrelated work. At the audit snapshot, 12
   manifest hashes were stale and three payload rows differed from an existing
   source, including the two MDL-writer copies. The focused payload integrity
   test fails. These counts describe this mutable checkout, not a release build.

7. **P2 — The declared Python build backend cannot be imported.**
   `pyproject.toml:11` names `setuptools.backends.legacy:build`.
   Importing that configured module raises `ModuleNotFoundError` in the installed
   environment. PEP 517 installation/build through this configuration cannot
   begin. Select an actual supported backend and exercise isolated wheel creation;
   a successful native Visual Studio build does not cover this distribution path.

8. **P2 — Optional export outputs are still mandatory during promotion.**
   `export_job.py:355` allows an absent output when `required=False`, but the
   promotion loop at line 213 unconditionally moves/links every output.
   A writer producing the required file and omitting an optional sidecar fails
   and rolls back the required file. Skip absent optional outputs during
   promotion and report only published paths. Reproduced through the public
   helper; no current production caller using this optional flag was established.

## Evidence and coverage

- [reproduce.py](reproduce.py) safely recreates seven fault scenarios in temporary
  directories or process-local fakes. [observations.json](observations.json)
  records the original audit run. These probes describe defects; they are not
  acceptance tests asserting correct behavior.
- Existing focused checks: **73 passed, 1 failed**, with 21 export cases excluded
  by the explicit selection. The failure was
  `test_python_payload_copies_are_byte_identical_and_manifested`.
- First run: selected export-job tests, resource-manager revision concurrency,
  payload byte/hash identity and duplicate renderer-contract identity: 19 passed,
  1 failed. Pytest also reported an existing cache-path warning.
- Second run, cache disabled: project model, gizmo/KMAX transforms, source fallback
  main-window import, Unreal import and GPU-culling regressions: 54 passed.
- Called the repository's MCP `compare_model_pipelines('k1', 'PLC_bench')`
  helper against installed game data: match, three nodes, 214 vertices and 112
  faces. Its comparison is summary-level; it does not prove full vertex,
  deformation or rendered parity. No broad model scan was run.
- Native Debug interaction, themes, viewport rendering, animation playback,
  retail traversal and save/reload were **not** tested in this audit. Source-mode
  import and backend probes are not substituted for those gates.
- Additional backend inconsistency: `KMaxSceneManager.clear_scene()` leaves
  cached camera/light collections populated until a later synchronization.
  Reproduced as zero objects, one camera and one light. UI impact was not
  established, so this is a follow-up rather than a retained visible defect.

## Consequences for the requested Unreal pipeline

Use revision-bound immutable import previews, one authoritative module owner,
project-local assets and recoverable transactions. Keep source components and
local edits separate from compiled sector grouping. The accepted Rhen Var
compiler and retail evidence remain a regression baseline, not proof of generic
Unreal conversion. Implement against the workflow plan and the forwarded brief;
do not absorb unrelated audit fixes into the pipeline silently.
