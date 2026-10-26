# Ghost Studio source publication — 2026-10-26

Owner: LordVaderCW.

The branch `codex/publish-ghost-studio-20261026` captures the latest local source
from the primary Ghost Studio checkout and includes GitHub's default branch at
`5dc8544b`. It contains the accumulated authoring, export, Unreal integration,
lighting, secondary-motion, automation, documentation and regression work.

The source selection contained 364 files, including 25 previously ignored
regression tests. Generated game assets, runtime output, private configuration,
book libraries and nested Git worktrees were excluded. The primary checkout's
files, branch and index were preserved by preparing the publication in a
separate checkout.

The integration preserves the newer local level-export UV and texture behavior,
the upstream selected-room module-loading behavior, and upstream texture-preview
memory limits. Embedded Python package metadata was regenerated for all 18
packages, and the level-export embedded copy now matches its canonical source.
The MCP export fixture now implements the current raw-texture lookup API.

Validation covered 301 changed Python files with compilation only and 68 focused
tests covering level export, saved-scene MCP export, browser module-loading
contracts, native payload integrity/registration, Unreal pipeline payloads and
repository secret hygiene. No credential-pattern candidates were found in the
selected publication files.

This is a source checkpoint. Existing unfinished features and audit findings
remain documented in `CHANGES.md`, the September program audit and the Unreal
pipeline implementation notes. No binary release or version bump was created.
Visual Studio Debug UI/theme checks, native rebuilds, retail validation and
unfinished workflow acceptance were outside this publication task. The branch
should not be treated as a fully qualified release.
