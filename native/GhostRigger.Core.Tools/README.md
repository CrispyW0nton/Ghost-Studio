# GhostRigger.Core.Tools

Merged namespace package boundary.

Owner: LordVaderCW

Package-local Python ownership:

- `Python/src/systems/bas/character_presets.py` owns the read-only product
  workflow that resolves installed K1/K2 UTC characters into Body Attachment
  System body, head, texture, and equipment-layer references. Legacy UTCs may
  reference assets shipped only by the other installed game, so the resolver
  records body/head/equipment source-game provenance per layer instead of
  disguising a cross-game fallback. It is package local because
  `src/systems/bas/` currently contains only recipe assets and has no canonical
  Python implementation.
- Canonical UTC inventories may contain engine-only items on creature models
  with no compatible visual socket (for example a turret with no `rhand`). The
  preset workflow preserves those item/model references as non-visual recipe
  metadata and skips only the impossible visual graft; strict manual BAS builds
  continue to report a missing-socket error.
