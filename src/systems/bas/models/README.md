# Body Attachment System Model Recipes

This folder stores lightweight `.json` builds created by the Body Attachment
System (BAS). A build records how a preview model was assembled: the base body,
the attached head or equipment layers, the target sockets, layer transforms, and
the runtime rules needed to rebuild the same BAS layout.

The runtime contract for how BAS layers attach, animate, and stay isolated from
body skinning is documented in `../README.md`. Treat that contract as the base
for every BAS model recipe.

BAS recipes do not store copied mesh, animation, texture, MDL, or MDX data. They
reference source game assets by game and resref so the preview can be rebuilt
from the installed KOTOR resources.

## JSON Shape

Each recipe uses the `ghostrigger.bas.model` schema and currently stores:

- `game`: source game used for the body and attachments, such as `K1` or `K2`.
- `body`: the base body resref/name/supermodel.
- `mode`: either `headless_body` for the normal headless-body BAS workflow or
  `full_body` when the loaded character already owns the head and only equipment
  hooks should be used.
- `layers`: ordered BAS layers for `BODY`, `HEAD`, `MASK`, `GOGGLES`,
  `L. HAND`, `R. HAND`, `L. Weapon`, `BELT`, and `R. Wep`.
- `layers[].transform`: per-layer local position, rotation quaternion, and scale
  relative to the socket. These values are preserved so heads and equipment can
  later get alignment/tweak controls without changing the file format.
- `attachments`: compact slot-to-resref map for attached layers.
- `character_preset`: optional source UTC identity, display name, canonical
  equipped UTI references, per-layer K1/K2 source provenance, and resolved
  body/head texture overrides when the build came from Character Preset mode.
  It also retains the resolved slot models and identifies engine-only inventory
  whose creature skeleton has no compatible visual socket. Reopening the recipe
  restores the preset presentation and the same appearance textures.
- `runtime`: notes that attachments are socket followers, body animations stay
  owned by the body, and attachment skinning is isolated from the body palette.

Hand slots are sockets only. Actual weapon models belong in `left_weapon` and
`right_weapon`, which follow the `lhand` and `rhand` dummy/socket nodes.
Head equipment models belong in `mask` and `goggles`, which follow `MaskHook`
and `GoggleHook` on the attached head or full-body character. Belt models belong
in `belt`, which follows the waist socket `pelvis_g`.

## Generated Files

Files are named from the game, body resref, and attached layer resrefs, for
example:

```text
k1_p_carthbb_pmha01_w_blstrpstl_001.json
```

Use the BAS toolbox `Save Build` button to choose a model/build name. Opening
one of these `.json` files through the normal model-open flow reimports the BAS
build as its own composed preview model, while still resolving the source pieces
from the installed game assets.

Saving the same build name updates the same JSON file. These files are intended
to be human-readable and safe to diff.

## Character Preset Mode

Character Preset mode scans the effective UTC templates in both configured
KOTOR installations on a background worker. Selecting a character resolves the
body and texture through `appearance.2da` plus the equipped armor UTI, resolves
the detachable head through `heads.2da`, and resolves visible masks, weapons,
and belt models through equipped UTIs plus `baseitems.2da`. Full-body creatures
remain full-body BAS builds; they are not forced through the humanoid head
attachment path. Legacy K2 UTCs for K1 characters may borrow their canonical
body, head, or equipment from the configured K1 installation; every borrowed
layer is labelled and saved with its actual source game. Incomplete unused UTCs
that have no complete model in either installation are omitted, so every
selectable preset has a loadable body/head composition. The source game data
stays read-only. Some stock creature UTCs equip inventory items that the game
does not render because the creature model lacks the corresponding hand, mask,
or belt socket. Preset loading preserves those canonical items in metadata and
skips only their impossible visual attachment, so turrets, droids, and other
non-humanoid templates still load. Manually authored BAS compositions remain
strict and continue to report incompatible sockets.
