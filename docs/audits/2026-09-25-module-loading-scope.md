# Content Browser module loading scope

Date: 2026-09-25
Owner: LordVaderCW

The Content Browser now exposes Module loading above the asset list. Selected
room only is the default when no preference exists. Entire connected map opts
into the existing LYT room expansion and placement path. Both double-click and
the main-scene context actions honor the setting. It affects subsequent loads;
it does not remove objects already loaded in the scene.

Ownership: GhostRigger.Core.GUI.Display owns the control, signal wiring, and
short settings/load orchestration. Existing Scene services continue to own
module placement. The edited Python files have no root src counterparts in this
split-source checkout; they are package-local source, not divergent copies.
No game parser, geometry, module layout, export rule, theme token, or major
layout metric changed. The setting uses the existing JSON settings mechanism.

## Verification

- Before implementation, nine default-scope cases failed because full-map
  resolution was attempted; the preference-control test failed because the
  control did not exist. Two existing whole-map behavior cases passed.
- Final targeted run: 25 passed. Command:

  `python -m pytest tests/test_content_browser_module_loading.py tests/test_module_categories.py tests/test_native_python_payloads.py::test_python_payload_copies_are_byte_identical_and_manifested tests/test_content_browser_panel.py::test_content_browser_context_menu_splits_scene_level_and_asset_actions tests/test_content_browser_panel.py::test_content_browser_add_to_current_scene_uses_explicit_signal -q`

- Coverage includes selected-only routing, clear/add paths, explicit connected
  map opt-in, native Qt double-click signal routing, saved choice round-trip,
  open Settings snapshot preservation, and disk-save failure behavior.
- Broader targeted browser-file run: 59 passed, seven failed. All seven failures
  reproduced after executing the original HEAD panel source in the test process:
  the old primary-activation expectation, split-class visual-profile source
  inspection, four splash expectations, and progress-toast source inspection.
  None was weakened or edited for this change.
- GUI Display payload regenerated and byte-identity/manifest check passed.
- Native host Debug|x64 MSBuild succeeded. The real native Debug executable was
  launched with F5 from the active Visual Studio instance. Default theme and
  ModernGL startup were observed through accessibility/logs.
- Windows screenshot capture failed with FrameArrived/channel timeouts, even
  after refreshing the returned window. UIA click failed because coordinate
  input geometry was unavailable. Therefore this is not visible qualification:
  the dropdown, actual K2:001ebo1 room loads, and Default/Matrix/Droid/Dark/Light/
  Classic appearance still need a working native capture/input session.
- Pre-change MCP K2:001ebo1 comparison completed successfully on the original
  workspace backend. It is a backend check, not proof of this worktree's UI.

## Review and remaining checks

Inline simplification used the reuse, quality, and efficiency rubrics. No
structural changes were warranted; the control reuses existing Qt layout and
settings helpers. The local review MCP returned no loaded models, so no local
model review or substitute cloud worker ran.

Code review covered the diff, call sites, settings dialog lifecycle, original
single-room fallback, and tests. It identified the nonmodal Settings snapshot
overwrite case, which was fixed and covered before the final check. No known
code findings remain. Release readiness is pending the visible workflow checks
above; automated tests do not replace them. No claim is made that this fixes
every NPC/VFX crash or bounds total scene memory.
