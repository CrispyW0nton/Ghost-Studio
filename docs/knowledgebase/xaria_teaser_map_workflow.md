# Xaria teaser map workflow

`scripts/build_xaria_teaser_map.py` builds the independent KOTOR II module
`xartease`, titled **Xaria: Ichor in the Deep Grove**. It is a cinematic and
companion-presentation fixture, not a replacement for the `plcaa` recruitment
regression map.

## Design contract

- Target game: KOTOR II.
- Playable room: installed KOTOR II `402dxn/402dxna`, imported through the
  Environment Kit as `k2_402dxn_402dxna`.
- Navigation authority: the stock 360-vertex, 538-face `402dxna` WOK.
- Added forest dressing: eight visual-only Terrain Kit rooms with empty WOKs.
- Module root: `xartease`.
- Voice lookup: `module.ifo` stores numeric `Mod_VO_ID=997`. Private `xt_dlg`
  is silent; its guarded post-combat handoff launches production `xaria.dlg`,
  whose 48-line package resolves from the ResRef-derived folders
  `StreamVoice/997/xaria`, `StreamVoice/997/atton`, and
  `StreamVoice/997/kreia`. The package covers 24 encounter lines, seven lesson
  lines, fifteen post-recruit lines, and the Atton/Kreia recruitment
  reactions. Candidate validation requires the exact 48-file set.
- Entry: `(-7.0, -58.0, 2.651398)`.
- Return transition: `(1.5, -1.2, 9.648286)` to
  `402dxn/From_401DXN`.
- Asset policy: the room and forest dressing come from the user's installed
  KOTOR II library. The module contains private encounter templates, dialogue,
  item, and scripts, but it is deliberately **not standalone**: Xaria's loose
  model/texture/portrait files, global 2DA rows, and runtime patches must
  already be installed. No K1 asset bytes or copied retail textures are
  packaged.

The layout owns four named spatial zones: Lower Arrival Trail, Layered Forest
Switchback, Ichor Showcase Clearing, and North Dxun Exit. The two 2.4-metre
routes are stored in the KMAP spatial-design ledger and must pass Spatial Audit
before export.

## Three-beat combat staging

The clearing deliberately separates each target and camera angle:

1. **Miststep: Ambush** — `XT_Wraid_1` at
   `(5.0, -21.0, 10.013909)`, with a 1.75 m reserved arrival pocket at
   `(7.0, -23.0, 10.049328)`.
2. **Ichor Lightning** — `XT_Wraid_2` at
   `(5.0, -17.0, 9.974767)`, on a clean east-west hero lane.
3. **Ichor Drain** — `XT_Wraid_3` at
   `(3.0, -13.0, 9.981651)`, on a separate rising diagonal.

The module owns an isolated schema-11 combat controller. `xt_intro.utt` runs
`xt_start.ncs`, which calls `xt_begin`. The starter validates every actor,
commits private state `1`, and asks the stock `plc_invisible` director to start
`xt_dlg`. Heartbeat and click recovery reacquire that same director
conversation rather than launching combat under the gameplay camera.

`xt_dlg` is presentation-only: one silent, unskippable camera-`111` entry,
blank root `OnEnd`, `xt_b1`, a 30-second safety dwell, and no replies or links.
The node deliberately outlives the 18.40-second worst-case scripted timeline.
There is no queued `ActionPauseConversation` or automatic dialogue progression;
confirmed deaths select cameras `112`, `113`, and `114`, and the final branch
explicitly closes the director dialogue. There is no opening conversation.

The three deterministic beats use the same bundled production-effect wrappers:

- camera `111` establishes for 1.25 seconds before `xt_b1`;
- `xt_b1`: marks Wraid 1 with compact token `1`, invokes row-287
  `kxar_d_mamb`, performs the visible follow-through, and finalizes the target
  at 2.60 seconds;
- confirmed death holds camera `111` for 0.85 seconds, cuts to camera `112`,
  establishes for 1.25 seconds, then `xt_b2` marks Wraid 2 with token `2`,
  faces the live target, invokes row-290 `kxar_d_ilight`, and finalizes it at
  2.80 seconds;
- confirmed death holds camera `112` for 0.85 seconds, cuts to camera `113`,
  establishes for 1.25 seconds, then `xt_b3` marks Wraid 3 with token `3`,
  faces the live target, invokes row-291 `kxar_d_idrain`, and finalizes it at
  3.10 seconds.

Each finalizer clears Plot and MinOneHP, applies explicit `EffectDeath`, and
calls `xt_dead` directly. Wraid `ScriptDeath` fields are blank; creature event
dispatch cannot suppress progression. After proof mask `7`, `xt_dead` commits
state `2`, destroys the approach trigger, holds camera `113` for 0.85 seconds,
selects camera `114`, holds it for 2.25 seconds, executes `k_oei_endconv` on
the director at 3.10 seconds, and schedules `xt_post` at 3.35 seconds. The
private DLG never performs recruitment; production `xaria.dlg` owns the roster
and party-selection handoff.

All creature LocalNumbers stay in retail's documented byte slots `12..28`.
Schema is local `24`, ordered death proof is `25`, power proof is `26`, entry
path is `23`, and state is `28`. Local `21` is an explicit in-flight latch:
values `1/2/3` own active beats and `12/13` own the two camera holds.
Heartbeat and click recovery may dispatch an unproved beat only when that latch
is zero. This prevents the longer cinematic holds from clearing Xaria's action
queue and replaying an already-running cast.

Encounter-local schema version `11` has only states `0 waiting`, `1 running`,
and `2 finished`; there is no terminal failure state. The trigger remains
until all three deaths are proven. Running-state click and heartbeat paths can
recover the first unfinished beat when no beat owns the timeline. Loading an
older candidate resets only private locals and never downgrades production
`KPM_XARIA_STATE`.

The packed UTCs write faction ID `2` for `XT_Xaria` and neutral faction ID `5`
for all three private wraids, preventing ordinary perception from stealing the
scene. Each beat changes only its current target to Hostile 1. Xaria has the
normal `Conversation=xaria` and `ScriptDialogue=xt_click`; clicking cannot
bypass unfinished combat, but it can recover a missed start or reopen declined
post-combat dialogue. If production Xaria is already recruited, this
independent teaser may still run for presentation, but the post hook never
downgrades committed global state `3`.

The build clones the installed fixtures only as source material, then rewrites
module-contained `xt_xaria.utc`, three wraid UTCs, `xt_intro.utt`,
`xt_dlg.dlg`, `xt_blade.uti`, the private invisible director, and the private
controller scripts. It also bundles exactly three hash-pinned production
wrappers: `kxar_d_mamb.ncs`, `kxar_d_ilight.ncs`, and
`kxar_d_idrain.ncs`. All inherited production event scripts are cleared before
export.

### Trigger coordinate contract

Odyssey stores a GIT trigger's `Geometry` vertices as offsets local to the
trigger's `Position`, not as world coordinates. The failed candidate supplied
world coordinates and the engine translated them a second time, placing the
real encounter volume away from the visible trail.

The encounter trigger has Position `(-5.625, -22.25, 9.929297798)` and these four
local vertices:

- `(-13.375, -5.25, 0.079067814)`
- `(9.625, -5.25, 0.282852481)`
- `(9.625, 5.25, 0.042318766)`
- `(-5.875, 5.25, -0.334007284)`

The exit has Position `(1.5, -1.2, 9.648286)` and local rectangle
`(-1.5,-0.55,0)`, `(1.5,-0.55,0)`, `(1.5,0.55,0)`,
`(-1.5,0.55,0)`. Package readback requires a near-zero local center and
records the derived world center as `Position + local center`. Never copy
world-space corner coordinates into a GIT `Geometry` list.

Six GIT cameras are present, but the approach itself never activates a
cinematic camera. The DLG activates the three combat cameras only as their
beats begin, then uses an eye-level Xaria close-up and player reverse after
combat. Their orientation is exported in Odyssey's GFF `w,x,y,z` order and
their pitch is stored in degrees, matching stock K1/K2 GIT camera records. The
last two shots are explicit checks for facial animation, emissive eyes, front
and rear hair motion, hair opacity, and dialogue framing.

The first failed retail run also exposed a separate environment problem. Map
Studio's Environment Kit intentionally stripped all source lightmap names and
secondary UVs, while its two authored lights were viewport metadata rather
than serialized MDL light nodes. Under the close green fog this rendered as a
green void over nearly black geometry. The teaser builder now restores only
the stock `402dxna_lm0` name and UV2 data after proving every donor and target
surface still has the same name and vertex count. Eight lightmapped surfaces
survive binary MDL/MDX readback; authored geometry, diffuse UVs, and collision
remain unchanged.

## Manual Ghost Studio recreation

1. Open **Map Studio**, create a **KOTOR II** project with module root
   `xartease`, and connect the KOTOR II resource library.
2. In **Environment Kit**, filter for `402dxn`; place
   `k2_402dxn_402dxna` at the world origin.
3. Set the player start to `(-7.0, -58.0, 2.651398)` and face it toward the
   first switchback point `(-13.25, -52.0)`.
4. Recreate the four zones and both 2.4 m routes from
   `artifacts/xaria_teaser/xaria_teaser_manifest.json`, then run
   **Spatial Audit**.
5. In **Terrain Kit**, place the eight manifest rows at their exact position,
   rotation, and scale. Keep every piece visual-only; do not generate
   collision.
6. In **Environment → World Lighting**, choose **Custom** and enter the
   manifest's ambient, diffuse, shadow, and green forest-fog values.
7. Place `xt_xaria`, `xt_wr1`, `xt_wr2`, and `xt_wr3` at the recorded
   positions. Place `xt_intro` across the approach at
   `(-5.625, -22.25, 9.929297798)` and enter the four **local** Geometry offsets
   from the trigger-coordinate contract above. Do not enter their world-space
   sums. Author `xt_xaria` with faction ID 2 and all three wraids with neutral
   faction ID 5. The first beat changes the wraids to Hostile 1 only after the
   cinematic is in control.
8. Place stock `plc_invisible` beside the trigger at the same position. It
   owns presentation dialogue `xt_dlg`. Give `xt_xaria`
   `Conversation=xaria` and `ScriptDialogue=xt_click`; that click script must
   enforce the private state/proof/in-flight gates before any production
   conversation can open.
9. Leave the Miststep arrival pocket empty. Do not put a camera, prop, or
   creature inside its 1.75 m reservation.
10. Add Camera IDs `110` through `115` using the manifest position, orientation,
   FOV, height, mic range, and pitch.
11. Build `xt_dlg` as one silent, unskippable camera-`111` entry with `xt_b1`,
    delay `30`, blank root `OnEnd`, and no links or replies. Do not queue
    `ActionPauseConversation` or `ActionResumeConversation`. Leave every wraid
    OnDeath script blank. The three beat finalizers call `xt_dead` directly:
    hold the outgoing shot for 0.85 seconds, cut to cameras `112` and `113`,
    establish each for 1.25 seconds, then run the next beat. After death three,
    hold camera `113` for 0.85 seconds, cut to `114`, hold for 2.25 seconds,
    execute `k_oei_endconv` on the director, then run guarded `xt_post`.
    Reserve LocalNumber `21` for active/scheduled beat ownership so click and
    heartbeat recovery cannot replay a cast. Package the exact row-287/290/291
    director wrappers. Every beat must clear Plot and MinOneHP and apply
    explicit `EffectDeath`; guarded `xt_post` starts production `xaria.dlg`
    independently after the presentation releases.
12. Add `newtransition` at the north threshold and configure:
    `LinkedTo=From_401DXN`, `LinkedToModule=402dxn`, `LinkedToFlags=2`, and
    `TransitionDestin=129401`. Its four Geometry points are also local to the
    transition Position.
13. In **Walkmesh**, paint threshold faces `102`, `103`, `106`, and `107` as
    surface `18` (`DOOR`). This is required for linked-transition readiness.
14. Run PIE from entry to the hero clearing, then from the clearing to the
    north exit. Run Spatial Audit and export/readback before saving the
    candidate.

The current Environment Kit UI does not yet expose a preserve-source-lightmaps
switch. A direct manual export therefore is not lighting-equivalent to the
candidate. The deterministic builder performs the validated eight-surface
restoration after the KMAP snapshot; do not claim a direct UI export has the
same retail lighting until that control exists.

The generated
`artifacts/xaria_teaser/GHOST_STUDIO_MANUAL_WORKFLOW.md` expands the exact prop
and camera rows into a click-by-click checklist.

## Explicit non-standalone dependency contract

The private encounter resources above are bundled. The following remain
external and must be verified by the final transactional staging manifest:

- Loose model/texture files: `p_xariabb.mdl`, `p_xariabb.mdx`,
  `p_xariab1.tga`, verified facial-performance head
  `p_xariah6.mdl`/`p_xariah6.mdx`, authored head texture
  `p_xaria06.tga`/`p_xaria06.txi`, and `po_pxaria.tga`.
- Forty-eight production WAV/LIP pairs. The original nine encounter identities
  use `997xaria001..997xaria009`, the seven lesson identities use
  `997xaria010..997xaria016`, and the fifteen expanded-meeting identities
  `xv_warn` through `xv_withhold` use `997xaria017..997xaria031`. The fifteen
  post-recruit identities remain `997xaria101..997xaria115`; the two
  companion reactions remain `997atton001` and `997kreia001`.
- Runtime WAV directory:
  `StreamVoice/997/<speaker>/<runtime-resref>.wav`.
- Runtime LIP files:
  `Override/<runtime-resref>.lip`.
- Global rows: `appearance.2da` row 725, `heads.2da` row 199,
  `portraits.2da` row 64, `classes.2da` row 17, and `spells.2da` rows
  287/290/291.
- Runtime behavior: `XariaPowerRuntime`'s action-862 hook and
  `CustomClassExtension`.
- The installed KOTOR II resource library for the room, stock supermodel,
  equipment, textures, and sounds.

The packed dialogue voice contract is exact:

- `xv_intro`: “Keep away, outlander. This valley will be your doom if you step
  a foot closer. Their fear is still moving through the ichor—and something in
  it recognizes you.”
- `xv_name`: “Xaria. I belonged to a clan once, then an army, then the dead.”

Both entries use speaker `XT_Xaria`, empty `Sound`, `SoundExists=1`, and zero
plot XP in production `xaria.dlg`. They map to `997xaria001` and
`997xaria002`. The original encounter continues through `997xaria009`, lessons
use `997xaria010..997xaria016`, and the expanded meeting uses
`997xaria017..997xaria031`; post-recruit and reaction lines retain their
separate ranges above. Because the IFO's voice ID is `997` and the production
DLG ResRef is `xaria`, retail derives each speaker folder from the runtime
ResRef. The Patch Manager stager validates the packed IFO,
production-dialogue dependency, and all 48 exact WAV/LIP pairs before
installing them as one recoverable transaction. Logical `xv_*` source paths
alone are not valid retail lookups.

The generated proof records the exact SHA-256 and size of every loose external
file found at build time. It leaves the global-row and runtime-patch entries
unverified until the final staging workflow can prove them; this is intentional
and must not be promoted to retail evidence.

## Evidence and remaining gates

### Schema-4 retail failure: cameras without combat, followed by a loop

The schema-4 module
`28B85260FABB7087E8D24E0600CA0CD3FB2A417F552F132EBE5D53AFA116F56A`
is a failed retail candidate. In observer session
`20260726-134624-custom-animation-flurry-plcaa`, the user visibly confirmed
that cameras 111–113 appeared, but Xaria and all three wraids remained in
`pause1`/`cpause1`; no attack, spell animation, impact, or death occurred.
The empty reply path could be clicked through and the failed end state reset
itself, so the same camera sequence repeated. Video
`Kotor-Patch-Manager/.tmp_ue_anim/evidence/xaria_schema4_full_encounter.mp4`
and its contact sheet preserve that failure. Camera motion is not combat
proof.

The initial diagnosis blamed the active cinematic DLG for suppressing the
actor queue. That was incomplete. Later packed-bytecode analysis found that
schema 4 and schema 5 both read and wrote out-of-range local Number slots, so
neither run ever proved that its combat node reached the action boundary.
Retain the visible observation above, but do not cite it as proof that
`CutsceneAttack` is blocked by an active DLG. It did prove that a failed
cinematic must never make itself retryable in-place: retry concealed the first
boundary failure and created the visible loop.

Installed KOTOR II assets provide the replacement contract:

- `303NAR_dlg.erf::hk50.dlg` entry 0 runs `a_hk50cut` during a four-second
  active node; the script delays and assigns `CutsceneAttack` without pausing
  the conversation.
- `301NAR_dlg.erf::thugboss.dlg` uses the same no-pause
  `CutsceneAttack` pattern during a four-second placed-camera node.
- `101PER_dlg.erf::admlog.dlg` delays a direct `ExecuteScript` while its
  dialogue remains active.
- `006EBO_dlg.erf::attonend.dlg` uses genuinely empty, finite-delay,
  individually unskippable camera entries. Whitespace text is unnecessary.

`CutsceneAttack` consumes an `animations.2da` row, not an `ACTION_*` constant.
The current blade beat uses row 135 (`c3a1`, two-handed attack); the two power
beats use row 62 (`castout1`). Passing `ACTION_CASTSPELL=4` would instead play
row 4 (`runinj`) and is forbidden by focused tests.

### Schema-5 retail failure: invalid local Number storage

Schema 5 replaced the action queue with the one-shot director documented
above, but it still failed in retail. The user visibly saw cameras 111–114
cycle while Xaria never attacked any wraid. The live module was
`166A87BE5F82F489CDC5E5643D15A5644F41FCBBFE30F3760405566947CCBB73`.

The packed DLG is finite and acyclic:
`E0 → R0 → E1 → R1 → E2 → R2 → E3 → END`. Its first three entries correctly
bind `xt_b1`, `xt_b2`, and `xt_b3`, and root `OnEnd` is `xt_post`. Vanilla K2
controls prove that this attachment is valid: `303nar/hk50.dlg` invokes
`a_hk50cut` from `Entry.Script`, and that script performs `CutsceneAttack`
during the active camera dialogue.

The actual defect was the local-variable contract. KOTOR II's own
`nwscript.nss` documents creature local Number indices `12..28` and stored
values `0..255`. Schema 5 used indices `60/61/62/64/66/67/68` and stored
`-2/287/290/291`. Packed ordering closes the causal chain:

- `xt_b1` read invalid state slot 61 before faction changes, the production
  wrapper, or `CutsceneAttack`, so it returned without combat;
- `xt_hb` read invalid schema/state slots 66/61 as unset and invoked
  `xt_begin` again;
- the DLG itself did not loop; the heartbeat externally restarted it.

The old schema-5 candidate record is retained below as deterministic
structural evidence, not a runtime pass:

- semantic digest:
  `352fe3caedca7e7942c5951fb32d158a109faebb34b2c4c11ef3e1ef49a5bd80`;
- `xartease.mod`: 1,238,619 bytes, SHA-256
  `166A87BE5F82F489CDC5E5643D15A5644F41FCBBFE30F3760405566947CCBB73`;
- `xartease.kmap`: 1,030,664 bytes, SHA-256
  `1DD3C00DE21C22D4253E34DD808F412BB023533209CFA430C659BBDFE8BDB56E`;
- `structural_proof.json`: 79,597 bytes, SHA-256
  `717307C9CBC6ADF3B4D85B5167317D96A5ACF612117B0A6EBACEE31F78B43234`;
- `xaria_teaser_manifest.json`: 77,406 bytes, SHA-256
  `12B174F6EB1B864039C060EF13165B4017EBE40903060D66F027EE14AA1C55B0`.

Package readback proves all 21 encounter-owned resources, including the three
director wrappers. Their compiled SHA-256 values are:

- `kxar_d_mamb.ncs`:
  `C94A94FD89109477D6A4C798AFF1BE99E18C8A5991A875F3C610F93CD7F5A5DB`;
- `kxar_d_ilight.ncs`:
  `4895B3BB6D50FF1A362B05F79E8469D658BC6264CC32E3F446F17AC1CF41B4E8`;
- `kxar_d_idrain.ncs`:
  `46D67A6ADAAFE813AC5EA481F8E875F982D85516C7C7D082031C6CAB81542677`.

The focused Ghost suite passes 16 tests and Ruff is clean. This is
deterministic structural evidence only.

Closed-game Patch Manager transaction
`20260726T152548505146` installed that exact module after 201 live dependency
records passed rehash validation. Its 59,138-byte stage manifest has SHA-256
`46BBD92F15224BEEA000FC5962408963B5CCCB8C8C077178EBF0698FAEB6F6A2`;
the transaction state is `committed`, and the installed module hash is
`166A87BE...`. The failed schema-4 module `28B85260...` is preserved as the
verified pre-install backup. The same manifest honestly records runtime proof
as `not tested in game`.

Schema 5 is a confirmed runtime failure and must not be staged again.

### Current schema-6 candidate

Schema 6 remaps the complete encounter state machine into retail-safe storage:

- target marker local 22;
- entry/schema/sequence/proof/watchdog/state locals 23–28;
- positive terminal failure state 99;
- compact effect/proof tokens `1/2/3` for Miststep, Lightning, and Drain;
- real spell rows `287/290/291` remain only as `ExecuteScript` run vars.

The shared Patch Manager power pack uses the same target/proof contract. Its
builder rejects every local Number index outside `12..28`, every literal
stored value outside `0..255`, and any generated local call it cannot verify.
The Ghost builder applies the same guard to every private encounter script.

Two clean Ghost builds produced the same 54-resource semantic state:

- semantic digest:
  `cf1af671f70cd6e2d518ab6f5f1966760ffef1785f9ac8218437a73c9ccc76f5`;
- `xartease.mod`: 1,238,666 bytes, SHA-256
  `89D6832D72F8D83F22287B51FC6A0DD57871BF9CC62F95678BD253A69D0E02B9`;
- `xartease.kmap`: 1,030,664 bytes, SHA-256
  `A3E17DA49F3711B33E2D17CB5ABE1EDD5D143C7FC7A311662EA08C948D418DA1`;
- `structural_proof.json`: 79,597 bytes, SHA-256
  `A779873E62F41BC19C6C7E909D993B4A98709BC9C4E6EA7FB2DB3B194E1C48D7`;
- `xaria_teaser_manifest.json`: 77,406 bytes, SHA-256
  `5CD7B79BE48ABA9A439FF30C58AECEBB9E5393DF7728ACA24E1C224B91D5F734`.

The rebuilt director wrappers are:

- `kxar_d_mamb.ncs`:
  `DDE03EC16FC2E46813E73500251F839BC30E4DDC13859D40C2A8088452E6DC15`;
- `kxar_d_ilight.ncs`:
  `12F7D40E08BAE376F0FD692D1E941C379287AD53909D2E134488F9C9BF42E49A`;
- `kxar_d_idrain.ncs`:
  `2FDDC015E06AA8E17FC84A9A88CBDA03E06EEB0917280A9887C2493855227CC3`.

Focused verification is green: 20 Ghost teaser tests and 33 coordinated Patch
Manager power/stager tests. Packed NCS readback confirms that all local Number
indices are 22–28, all stored literals fit a byte, and the real spell rows
remain intact as script vars. This is deterministic structural evidence only.

Closed-game Patch Manager transaction
`20260726T183613495671` committed this exact schema-6 module and all 17
runtime voice copies after 201 live dependency records passed validation.
Its 59,138-byte stage manifest has SHA-256
`59D94C425CDFAE2919226B102055E1E27468424807AA40C787908C99885208D6`;
the installed `Modules/xartease.mod` is 1,238,666 bytes and exactly matches
candidate SHA-256
`89D6832D72F8D83F22287B51FC6A0DD57871BF9CC62F95678BD253A69D0E02B9`.
Installed `kxar_d_mamb.ncs`, `kxar_d_ilight.ncs`, and
`kxar_d_idrain.ncs` also match the corrected hashes above. Transactional
installation is proven; runtime proof remains `not tested in game`. The next
retail-visible test must still prove all three attacks and deaths, terminal
post-fight dialogue, and absence of a restart loop.

The older schema-3 timing and staging record below is retained as failed
history. Its skippable/retry contract is superseded and must not be copied
into a new build.

### Retail failure that established the timing contract

Candidate
`EB0286305F793BCF1C04C33647FA7E382F2F938550FF594E76A8EBF54BB38445`
visibly produced only a one-frame camera flicker, no combat, and no response
when Xaria was clicked afterward. Observer session
`20260726-115432-custom-animation-flurry-plcaa` recorded two `pause1` events
about 63 ms apart and no cast or attack animation. Every silent DLG entry in
that candidate had the default `Delay=-1`, so its unconditional textless graph
could finish before the queued pause action took control. The starter had also
latched state `1`; neither heartbeat nor click could recover that state.

Installed vanilla evidence established that explicit dwell is required:

- K1 `kas_m24aa_s.rim::kas24_jolee_01.dlg`, SHA-256
  `5086E944855F7452BFD8A01743673CB580F98B23A465DE591F41F32FAE0C0DCA`,
  uses silent-camera delays `5/3/3`;
- K2 `106per.mod::106droid.dlg` uses silent-camera delays `3/3` around
  `ActionPauseConversation`.

The current contract is therefore `Skippable=1`, entry delays `5/4/4/1`,
entry scripts `xt_b1/xt_b2/xt_b3/<blank>`, and root `OnEnd=xt_post`.
Encounter schema `3` migrates stale schema-2 state, and rejected handoffs enter
state `-1`, which `xt_click` can reset and retry through `xt_begin`. These are
structural corrections; their retail-visible success remains pending.

The builder runs twice in clean temporary roots and compares the sorted
resource-content manifest. The current 51-resource semantic digest is
`0cdb5c8a07a50e089d08a46e23b9d9b05b43d9e1b4e373bd0a70671eb9d27f26`.
The candidate MOD is 1,231,958 bytes with SHA-256
`024E0AE430533A674F5C87953FD803EB76C1FC15EEA8C1699AF03E726D1E2026`.
The 78,642-byte structural proof SHA-256 is
`EA1B19786BE1A2FE5975BF9C35307F538EEE3A0497D1D4571DDFA36EADD7D346`;
the 75,818-byte teaser manifest SHA-256 is
`FCEFA27D8CCE852CC6D081E3D3624B1764754C7B594D9F930B96C93E622719D2`.

Package readback parses ARE/GIT/IFO/PTH, all nine WOKs, four creature resrefs,
both triggers, stock `plc_invisible`, camera IDs 110–115 with exact
orientation/pitch values, all 18 private encounter resources, and all ten
NCS headers. It proves:

- both trigger Geometry lists are local to their Position and derive the
  intended world centers;
- `Mod_VO_ID=xar`, module folder `xar`, and dialogue ResRef `xt_dlg`;
- private Xaria has tag `XT_Xaria`, heartbeat `xt_hb`, and a blank
  Conversation field, plus appearance 725, portrait 64, class 17, and packed
  faction ID 2;
- `XT_Wraid_1`, `XT_Wraid_2`, and `XT_Wraid_3` each have packed faction ID 1;
- `xt_intro` runs `xt_start` and its local Geometry center is
  `[0, 0, 0.0069395]`;
- the exact non-looping four-entry/three-reply private DLG graph, with camera
  IDs `111/112/113/114`, scripts `xt_b1/xt_b2/xt_b3/<blank>`, delays
  `5/4/4/1`, `Skippable=1`, and root `OnEnd=xt_post`;
- the director and all three row-287/290/291 impact scripts include the
  local-64 proof contract, schema version `3`, fail-closed `xt_post` handoff,
  and explicit state-`-1` click retry;
- all six cameras, IDs 110–115;
- eight `402dxna_lm0` surfaces whose UV2 counts equal their vertex counts.

An explicit negative inventory check finds none of the production
`xar_*`/`kxar_*` encounter resources. PIE reaches both destinations without a
pathing blocker. Independent installed readback passed every boundary above
and matched the current hash-bound external dependency contract. Focused Ghost
and Patch Manager tests and Ruff validation passed before closed-game staging.

The final closed-game installation session
`20260726T124759383864` committed the exact module and 17 runtime WAV copies
after 195 live dependencies passed validation. Its stage manifest has SHA-256
`17F6B5F9202538423F7CBE26293F04EBE991B1AF70DD82AAF077D5AF3457078C`.
The prior timed-but-schema-2 module
`2088D66C4D2CBA9F4541DC074E2D083812BFECD39D756C3DFA0CCD26A3B62DEE`
was preserved as the verified backup. The stage manifest explicitly records
runtime proof as `not tested in game`; installation is proven, retail behavior
is not.

One staging retry exposed a representation-only hash mismatch: Ghost evidence
used lowercase hexadecimal while the Patch Manager's local recorder used
uppercase. SHA-256 validation now normalizes hexadecimal case before comparing
while still rejecting a non-64-character or non-hex value. A regression fixture
keeps lowercase evidence so this cannot return.

The canonical spatial-design source and both native payload copies are
byte-identical at SHA-256
`fd3b08409c625a8849e2de73476f62901ef70709c23d3e01368085327d9c4423`;
their package bridge and manifest rows are also checked directly. A full
package-wide payload-generator run was not used for this handoff because
unrelated in-progress Rhen Var work leaves `authored_skybox.py` intentionally
different from its native payload copies. Do not normalize that unrelated
work; rerun the full generator only after its owner synchronizes it.

The blueprint and storyboard PNGs are structural evidence only. They are not
retail-game proof.

Retail acceptance still requires:

- Install through the transactional patch workflow while KOTOR II is closed.
- `warp xartease` and visibly traverse entry → clearing → exit.
- Confirm approach starts Miststep and camera 111 together without an intro
  conversation.
- Confirm cameras 111–113 activate with Miststep, Lightning, and Drain,
  respectively, and each target dies before the next shot.
- Confirm Xaria's first spoken line and camera 114 occur only after all three
  wraids are dead; audibly confirm both Voice Design 1 lines play and visibly
  confirm their facial animation; confirm the reply chain does not loop.
- Confirm the restored room is readable forest geometry rather than green fog
  over a black void.
- Enter with production Xaria already recruited; confirm the private teaser
  still starts and the production roster/recruitment state is unchanged.
- Confirm Xaria's front and rear hair physics, emissive eyes, green mist,
  facial animation, dialogue close-ups, and blade presentation.
- Confirm the exit arrives at the stock `From_401DXN` waypoint.

## 2026-07-26 schema-7 death-driven combat correction

Retail session
`KotorDebugger/sessions/20260726-184253-custom-animation-flurry-plcaa`
invalidated schema 6. At `18:43:33`, `xt_begin` produced exactly three
`cdamages` events by setting MinOneHP and applying 1000 setup damage. That
parked all wraids at 1 HP before any power landed. The four fixed DLG entries
then occupied 14 seconds. Xaria's first non-idle animation (`runST`) did not
begin until `18:43:47`, when the dialogue ended; no cast, attack, later damage,
or `cdead` event occurred. The visible 1-HP bars were setup damage, not power
proof.

Two assumptions are now explicitly rejected:

- `AssignCommand(oXaria, CutsceneAttack(...))` does **not** run while an
  unpaused cinematic DLG owns the scene, even when an invisible placeable owns
  the dialogue.
- A production helper proof token must not gate cinematic cleanup or
  progression. Miststep can legitimately reject randomized destination
  sampling, but a rejected optional helper cannot leave a protected target
  alive or suppress the remaining showcase.

The replacement follows the installed K1 Jolee encounter and K2 Sion death
pattern documented by the community:

- [Jolee pauses dialogue before combat](https://github.com/KOTORCommunityPatches/Vanilla_KOTOR_Script_Source/blob/master/K1/Modules/M24AA_Kashyyyk_Upper_Shadowlands_kas_m24aa/k_pkas_joleecut3.nss);
- [Jolee advances cameras from creature events](https://github.com/KOTORCommunityPatches/Vanilla_KOTOR_Script_Source/blob/master/K1/Modules/M24AA_Kashyyyk_Upper_Shadowlands_kas_m24aa/k_pkas_katarnuse.nss);
- [Deadly Stream scripted-fight guidance](https://deadlystream.com/topic/3569-making-ingame-cutscenes/) says to hold actors with MinOneHP only until the intended impact, then clear it and explicitly kill;
- [the Sion cutscene discussion](https://deadlystream.com/topic/12196-how-do-i-extend-the-duration-of-a-cutscene-and-destory-the-remains-of-an-npc-during-said-extended-cutscene/?comment=100868&do=findComment) records the same Plot/MinOneHP/EffectDeath finalizer.

Schema 7 therefore:

- removes all pre-combat 1000-damage setup;
- uses one non-skippable camera-111 entry and immediately pauses its
  director-owned conversation;
- runs Xaria's freed action queue while the placeable camera remains active;
- calls the row-287/290/291 production wrappers, but uses the same authored
  green VFX as a non-blocking fallback if a helper rejects;
- clears Plot and MinOneHP and applies `EffectDeath(FALSE, FALSE, FALSE)` at
  each intended impact;
- gives all three wraids `xt_dead` as OnDeath, switching cameras
  `111 → 112 → 113 → 114`, dispatching beats two and three, and resuming the
  conversation only after the third confirmed death;
- keeps heartbeat advancement idempotent and retains the one-shot trigger,
  failure latch, recruitment handoff, and origin cleanup.

Focused verification passed 20 tests and Ruff. Two clean builds matched, all
55 semantic resources passed packed readback, and the one-entry DLG graph
validated. The candidate evidence is:

- `xartease.mod`: 1,234,948 bytes, SHA-256
  `6ECF88766E9E939C882E21236FC047AA9788AD06CE4B12442994A51CE3CB88B5`;
- `xartease.kmap`: 1,030,854 bytes, SHA-256
  `70C263F583FCE768490ACABDFD1DE392E58333946A2E236A7884524276CC7643`;
- `structural_proof.json`: 79,814 bytes, SHA-256
  `7AF2E0B9E5F952C690D2BCC4E227275C8405739246EA37B42DFD697AFB34FDD5`;
- `xaria_teaser_manifest.json`: 77,975 bytes, SHA-256
  `C760111DA1FE0A5836D5D16C658864EE562CAA991DA40B32EAEA73403FB10FCB`;
- `xt_dead.ncs`: 972 bytes, SHA-256
  `32525257522FE0C99DC479D07EBEB217107527D7484144F60A37B5EA3AFD102C`.

These are structural/build results, not retail-visible success. The next test
must still visibly confirm that camera 111 and combat begin together, all
three power animations/effects play, each wraid dies before the next camera,
camera 114 leads once into voiced recruitment dialogue, and the encounter
does not restart.

The Patch Manager stager was tightened with the same schema-7 contract:
`xt_dead.ncs` is required and hash-bound, all three wraid UTCs must use it as
`ScriptDeath`, the single camera-111 entry must be silent, non-skippable, and
run `xt_b1` with `OnEnd=xt_post`, and the obsolete four-node timer spine is
rejected. The focused stager, voice-transaction, and recovery suite passed 27
tests, and Ruff passed.

KOTOR II was confirmed closed before the candidate was installed
transactionally. Session
`Kotor-Patch-Manager/logs/k2-xaria-teaser-staging/20260726T202306138552`
committed the exact 1,234,948-byte module after 201 live dependencies passed
rehash validation. The installed
`Modules/xartease.mod` has SHA-256
`6ECF88766E9E939C882E21236FC047AA9788AD06CE4B12442994A51CE3CB88B5`.
The prior schema-6 module was preserved as
`Modules/xartease.mod.before`, 1,238,666 bytes, SHA-256
`89D6832D72F8D83F22287B51FC6A0DD57871BF9CC62F95678BD253A69D0E02B9`.
The 59,138-byte stage manifest has SHA-256
`6A07AFB3B0116E5CE271ED7F0379B85ED12CFB8D105C9C8C3D3C7BB89A2D821A`
and honestly records `runtime_proof=not tested in game`.

## 2026-07-26 schema-7 retail failure and engine/community audit

The first schema-7 retail attempt showed neither a sustained camera nor
combat. The exact installed module was not stale:

- installed and active `xartease.mod`: 1,234,948 bytes, SHA-256
  `6ECF88766E9E939C882E21236FC047AA9788AD06CE4B12442994A51CE3CB88B5`;
- `swkotor2.exe` PID 85896 started at 22:05:50;
- the 22:06:11 autosave recorded `LASTMODULE=xartease`;
- the engine's `currentgame\xartease.mod` matched the installed module byte for
  byte.

Debugger session
`KotorDebugger/sessions/20260726-221128-custom-animation-flurry-plcaa`
attached at 22:11:29, 5 minutes 18 seconds after module entry. After attachment
Xaria's PFBNM body remained live at the authored position
`(-7.5061, -21.1558, 9.8227)`, but there were zero Xaria animation dispatches,
zero wraid death animations, zero Xaria power-runtime messages, and no
observed cast, damage, or death boundary. The session was configured for the
older `plcaa`/`pfbnm` observer and had no NWScript/local-variable probes, so it
does not prove whether `xt_start` failed to run or `xt_begin` latched the
encounter before the late attachment. It does prove that the custom powers
were never reached after attachment and that the failure was not a crash or
stale module.

Ghidra analysis of the Steam Aspyr executable corrected several assumptions:

- DLG Script1 dispatch occurs at `0x006CD43E`, after the entry/camera setup and
  before the dialogue advance call at `0x006CD544`.
- NWScript routines 205/206 map to `ActionPauseConversation` at `0x0066A950`
  and `ActionResumeConversation` at `0x0066AC60`.
- A valid living placeable is allowed to pause a conversation. The normal
  branch reaches `0x0066AAA1` and writes the pause field at
  `0x0066AAB4`. The staged `XT_Director` has 15 current and maximum HP, so the
  zero-HP/dead-placeable no-op branch is not implicated by its UTP.
- Script1 runs with the conversation owner as `OBJECT_SELF`. For
  `AssignCommand(XT_Director, ActionStartConversation(PC, ...))`, that owner is
  `XT_Director`; the PC is a separate target.
- object offset `+0xEC` is a default `CSWSObject` flag, not a conversation
  pointer. The per-owner conversation object is at `+0x34`.
- a one-node blank terminal DLG is not rejected by the loader merely because
  it is blank or terminal. The engine error about a last node containing an
  END or CONTINUE node applies to a mixed candidate-choice set, not this
  graph.

The decisive addresses for the next live probe are:

- `0x006CD43E`: Script1 dispatch;
- `0x0066A950`: pause routine entry;
- `0x0066AAB4`: normal pause-field write;
- `0x0054666C`: dialogue-manager pause check;
- `0x00546684`: pause accepted/wait path;
- `0x005466B8` and `0x005466D2`: advance/OnEnd path.

An installed-game census covered 10,608 NCS resources. It found 112 unique
scripts that invoke Action 205; every one also contains Action 206. Of 897
unique DLG-node reference signatures, 822 have successors and 75 are
terminal. Only five are blank terminal Entries. `xt_dlg` is uniquely the only
one-entry/zero-reply DLG combining a blank terminal camera Entry,
`Script1=xt_b1`, and root `OnEnd=xt_post`.

This invalidates the claim that schema 7 copied a proven vanilla topology.
The placeable and pause command are legal individually, but the combined
single-terminal-node controller is an invented, unproven edge case. Any
failure before the pause field is observed lets that same boundary invoke
`xt_post`; `xt_post` sets state 99 because no death proof exists, while
`xt_begin` has already destroyed `XT_Intro_Trigger`. The heartbeat and click
paths then refuse to retry. A transient entry failure therefore becomes a
silent permanent lockout.

The stock TSL pattern to reproduce is `851nih.mod` /
`851mand.dlg::a_openingcs`: linked, nonterminal camera nodes; a fixed delayed
resume for every pause branch; one controller for the timeline; actors made
commandable; action queues cleared; local boolean 87 used to suppress AI;
cinematic factions; MinOneHP; assured-hit attacks or explicit deaths; and a
separate terminal cleanup. The same general structure is described in
[Deadly Stream's in-game cutscene guidance](https://deadlystream.com/topic/3569-making-ingame-cutscenes/),
its [pause/resume troubleshooting thread](https://deadlystream.com/topic/4433-kotor-help-with-an-in-game-cutscene/page/3/),
and its
[trigger-started dialogue example](https://deadlystream.com/topic/4899-how-to-start-a-dialogue-when-the-player-reaches-a-point-in-an-area/).

There is a second independent defect behind the earlier visible “one attack
then stop” result: schema 7 queues an ordinary `ActionAttack` without the full
vanilla cutscene-actor setup. The stock scene makes actors commandable,
suppresses AI, controls factions, and uses deterministic assured-hit
talents/deaths. AI can otherwise reclaim Xaria's ordinary action queue.

No source or installed-game correction was made during this diagnostic pass.
The replacement must use a new schema so old local state is reset, retain a
retry path until entry proof exists, replace the terminal starter with a
linked vanilla-style controller, separate cleanup from the first node, and
arm the debugger before module entry with the addresses above. Structural
tests must reject the unique schema-7 topology rather than asserting it as a
requirement.

## 2026-07-27 schema-8 direct combat controller

The next retail report was a complete no-start: neither proximity entry nor
click produced combat or dialogue. A full installed-game trigger census and a
fresh architecture audit corrected the diagnosis before another iteration:

- `ScriptOnEnter` with blank `OnClick` is normal KOTOR II structure. Of 1,410
  installed UTTs inspected, 1,082 use a nonblank `ScriptOnEnter` and none use a
  nonblank `OnClick`; `xt_intro.utt` was not malformed.
- The local PyKotor compiler and `nwnnsscomp_k2.exe` produced the same
  instruction stream for the nested conversation launch. The compiler was not
  the cause.
- Schema 7 destroyed the only trigger and committed state 1 before the
  camera DLG proved it had started. State 1 was then a silent click no-op, and
  root `xt_post` could latch terminal state 99. A single missed presentation
  boundary therefore made the encounter permanently inert.
- The one-entry/zero-reply `xt_dlg` topology was unique in the installed
  dialogue corpus and did not reproduce the connected vanilla controller it
  was claimed to follow.

Schema 8 reduces the private encounter to states `0 waiting`, `1 running`, and
`2 finished`. It deliberately separates gameplay from presentation:

- `xt_start` and `xt_click` share `xt_begin`; private Xaria now has the normal
  `Conversation=xaria` plus `ScriptDialogue=xt_click`.
- `xt_begin` starts `xt_b1` synchronously before requesting any camera
  dialogue. Combat therefore has no delayed area-object or dialogue-start
  dependency.
- `xt_dlg` is a linked two-entry/one-reply shell. Entry 0 uses camera 111 and
  pauses through `xt_b1`; its blank bridge reaches camera 114 only after the
  confirmed third death resumes the conversation. Root `OnEnd` is blank.
- Confirmed deaths advance directly from `xt_dead` to `xt_b2` and `xt_b3`.
  Each beat clears Plot and MinOneHP and applies explicit `EffectDeath`; wraid
  `ScriptDeath` fields are blank and cannot gate progression.
- The trigger remains in the module until proof mask 7 confirms all three
  deaths. Every state-1 click and heartbeat retries the first unfinished,
  idempotent beat. There is no failure state.
- `xt_post` is scheduled independently after the third death and waits for
  the presentation dialogue to release before starting production
  `xaria.dlg`.

The native-ACTION verifier was updated for the new stack layout and now
asserts the synchronous `xt_begin -> xt_b1` call as well as both conversation
boundaries. Ghost Studio's 20 focused tests and Ruff pass. Two clean builds
matched; packed readback validates all 55 semantic resources, the linked DLG,
Xaria's conventional click contract, blank wraid death scripts, and schema-8
proof metadata.

Candidate evidence:

- `xartease.mod`: 1,237,017 bytes, SHA-256
  `B599BBF777DE8250C8E757EA8E264F0FC464AF878490FF507622CECE1AF00E3B`;
- `xartease.kmap`: 1,030,890 bytes, SHA-256
  `5A52942FE48F617C2503DB04680CE871A6C54C6374D3CB1A06D9DC4709BF2820`;
- `structural_proof.json`: 80,370 bytes, SHA-256
  `E7A8831BBE4FACD74EA9608A0537BD66E5CB3F43C6A77C2326F145E551F374AE`;
- `xaria_teaser_manifest.json`: 78,015 bytes, SHA-256
  `97121CCC9A80541A684A0BAAB68F15A6FBF34FC0FE5CC96AE3F5A4C2C260AD22`.

The Patch Manager stager now rejects every superseded schema-7 invariant. It
requires the schema-8 proof, Xaria's `Conversation=xaria` and
`ScriptDialogue=xt_click`, `xt_intro.ScriptOnEnter=xt_start`, the linked
camera `111 -> blank reply -> 114` shell, blank root `OnEnd`, independent
post-combat handoff, and blank wraid `ScriptDeath`. Its 32 focused tests and
Ruff pass.

KOTOR II was closed before transaction
`Kotor-Patch-Manager/logs/k2-xaria-teaser-staging/20260727T014150203208`
committed the exact candidate after rehashing 201 live dependencies. The
previous schema-7 module was preserved at
`Modules/xartease.mod.before`, 1,234,948 bytes, SHA-256
`6ECF88766E9E939C882E21236FC047AA9788AD06CE4B12442994A51CE3CB88B5`.
The 59,138-byte stage manifest has SHA-256
`A01EEEDF96206D488643831BB5A2D545E601C77DF4712ADD39F0468D17F03555`;
it records `install_state=committed`, 201 verified dependencies, and
`runtime_proof=not tested in game`.

This remains structural and installation proof, not visible runtime proof.
The next run must load a save outside `xartease` before warping in, because a
save made inside the old module can restore embedded schema-7 module state.
Arm the observer before entry. A successful start must immediately show the
message `Xaria encounter entry 101 started.` and begin Miststep before camera
presentation can become a gate.

## 2026-07-27 schema-8 presentation, voice, and live-anchor correction

The user's next retail run supplied the first positive schema-8 combat result:
the encounter started and reached combat. Preserve that gameplay architecture.
The presentation still failed in three separable ways:

- power cameras cut too quickly;
- caster effects appeared to remain at Xaria's original location after she
  moved;
- the post-fight subtitle played on a close-up, but Voice Design 1 was silent
  and Xaria's lips never separated.

Every one of the supplied MP4's 550 frames was reviewed. Frames 64–549 show
the speaking close-up; the lips remain sealed while cheek, nose, and mouth
surfaces deform. The audio track is effectively silent. Combat is absent from
that MP4, so the camera/VFX findings remain user-observed runtime reports.

Disassembly of KOTOR II's voice resolver at `0x006CC140` invalidated the old
alphabetic `xar/xv_*` lookup. Its nested folder branch accepts `gbl`, `avo`,
`000`, or a positive three-digit numeric prefix, then derives:

`StreamVoice/<resref[0:3]>/<resref[3:-3]>/<full-resref>`

The corrected contract is numeric ID `997`, Xaria dialogue directory `xaria`,
and runtime ResRefs `997xaria001..997xaria031` plus
`997xaria101..997xaria115`; Atton and Kreia use their matching speaker folders.
The IFO stores `Mod_VO_ID=997`; production DLG nodes bind those runtime ResRefs;
matching WAVs live under the ResRef-derived `StreamVoice/997/<speaker>` folder
and matching LIPs live in Override. Voiced nodes temporarily use
`FacialAnim=0` so the next test isolates the retail LIP track from the broad
expression overlays seen pinching Xaria's face.

The teaser presentation now holds camera 111 through the 2.60-second Miststep
beat, gives cameras 112 and 113 a 0.65-second establishing hold before their
2.80/3.10-second Lightning and Drain completions, settles camera 114 for 0.75
seconds, and delays the guarded production handoff to 1.65 seconds. This keeps
fixed cameras—there is no false claim of interpolated camera movement.

All caster-side mist calls now resolve from the live Xaria object. Miststep
applies departure mist before movement and an object-bound arrival mist after
the jump; Lightning and Drain face the live target and add caster mist at
Xaria's current position. Both powers now also use retail `EffectBeam` with
Xaria as the live effector, so the beam spine follows her current hand-to-target
geometry. The custom green impact model remains victim-bound and the programmed
beam core retains its stock color; no global stock texture is replaced.

The longer shots exposed a recovery race during review: heartbeat or click
could re-enter an unproved beat before its delayed death call completed,
clearing Xaria's action queue and duplicating the cast. LocalNumber `21` is now
an explicit owner latch (`1/2/3` active, `12/13` camera hold). Recovery runs
only when it is zero, and the post-combat handoff requires it cleared.

Verification completed without installing over the running game:

- Ghost focused teaser suite: 25 passed;
- coordinated Patch Manager voice/power/encounter/stager/recovery suite:
  63 passed;
- numeric PLCaa dry-run candidate and all 17 WAV/LIP pairs passed exact
  readback;
- both clean teaser builds matched all 71 prepared file records as well as the
  packed module digest;
- read-only Patch Manager validation accepted the complete 86-dependency
  teaser contract while KOTOR II remained open;
- no live KOTOR II file was replaced.

### Prepared-candidate build gate

Ghost Studio no longer requires the corrected package to be installed before
it can build the teaser. Set `XARIA_CANDIDATE_ROOT` to one KPM staging
session's `candidates` directory:

```powershell
$env:XARIA_CANDIDATE_ROOT = "C:\Users\NewAdmin\Documents\GDeveloper\Workspaces\Kotor-Patch-Manager\logs\k2-plcaa-xaria-staging\20260727T094053\candidates"
py scripts\build_xaria_teaser_map.py
Remove-Item Env:XARIA_CANDIDATE_ROOT
```

This is an all-or-nothing snapshot. The builder uses that candidate's
`Override` and `GameRoot` for every Xaria-owned loose dependency and validates
all of them against the sibling `stage-manifest.json`. It rejects a wrong
operation, anything other than the uninstalled `not_requested` state, missing
manifest records, hash/size drift, mixed live/candidate inputs, or a dependency
change between the two clean builds. Absolute source paths recorded inside the
KPM manifest are not trusted; candidate paths are derived beneath the supplied
root. The three bundled director wrappers are also read only from this selected
Override, never from the mutable power-build candidate directory.

The resulting structural artifacts are:

- `xartease.mod`: 1,238,480 bytes, SHA-256
  `650D7E48E08C5727E97CBC17D8F824B754F3119B520874DEDDEBD8F958B483EF`;
- `structural_proof.json`: 84,874 bytes, SHA-256
  `4901A9D5B0DB9C59640ED2C4659B3EAE68D2A93F1852E9C1C2600EC65095AB3D`;
- `xaria_teaser_manifest.json`: 82,240 bytes, SHA-256
  `6162D0C5DE22BBDA7F83C45B01D44A4E2B19D3840AEB1666B72F3674A25A70CF`.

`py tools\stage_k2_xaria_teaser_map.py` now performs candidate-only validation
while the game is running. `--install` still requires KOTOR II closed before
validation and at every transaction recheck. Recovery of an incomplete teaser
transaction repeats that check after acquiring the stage lock and immediately
before each live unlink or restore.

KOTOR II was then closed and both transactions committed:

- production session `20260727T101034` installed PLCaa SHA-256
  `AF6D009F9453C1A7851A4E8F28201ECF67B35D4E934B41FED53DD1C58366F040`
  plus all numeric voice/LIP files;
- teaser session `20260727T101102222702` rehashed 201 live dependencies and
  installed `xartease.mod` SHA-256
  `650D7E48E08C5727E97CBC17D8F824B754F3119B520874DEDDEBD8F958B483EF`;
- the post-audit rebuild, including snapshot-only director wrappers, produced
  the same module hash as the installed file, so no second replacement was
  necessary.

Retail acceptance still requires slower readable beats, effects at Xaria's
live position, audible VO, visible jaw opening and phoneme changes, no
encounter repeat, and the existing recruitment cleanup.

## 2026-07-27 schema-10 death-gated camera ownership correction

The next retail capture proved that schema 9 still allowed the private camera
shell to disappear almost immediately. Its first silent entry serialized
`Delay=0`, while `xt_b1` queued `ActionPauseConversation`; retail followed the
blank link to camera 114 before the queued pause owned the conversation.
Combat and death progression continued, but later calls to cameras 112 and 113
had no active placed-camera dialogue to control.

This repeats the previously recorded silent-dialogue race and is not a new
engine limitation. Installed vanilla controls use explicit dwell: K1 Jolee
uses `5/3/3` on its silent placed-camera sequence, and K2 `106droid.dlg` uses
`3/3`. Schema 10 therefore keeps the two-entry presentation shell but changes
its delays to `3/1`. Camera 111 now has time to become the paused owner before
beat 1. The existing `xt_dead` controller remains the only progression
authority: it refuses to commit a proof bit while the corresponding Wraid is
alive, cuts to 112 after confirmed death 1, to 113 after confirmed death 2,
and to 114 after confirmed death 3. Camera 114 now holds 1.50 seconds before
the director resumes.

The builder readback, Ghost tests, Patch Manager validator, and Patch Manager
tests all reject the superseded zero-delay/schema-9 contract. This is static
and automated evidence only; the next retail run must visibly confirm that all
three deaths occur under cameras 111–113 and the final camera-114 hold remains
on screen.

## 2026-07-27 schema-11 single-owner camera correction

The schema-10 retail run proved that a queued conversation pause was still not
a durable camera owner. The private dialogue naturally ended after roughly
four seconds, while the three combat beats continued for about eleven seconds.
Camera-selection calls made after that end had no active placed-camera
conversation to control. The same trace also showed death-driven cuts landing
only 0.20-0.45 seconds after the outgoing target died, which made the surviving
shots too brief to read.

Schema 11 removes the timing race instead of attempting another pause/resume
variation. `xt_dlg` now contains exactly one silent, unskippable camera-111
entry with `xt_b1`, a 30-second dwell, a blank root `OnEnd`, and no replies or
links. Neither the source nor the compiled private scripts use
`ActionPauseConversation`/routine 205 or
`ActionResumeConversation`/routine 206. The node safely exceeds the
18.40-second maximum scripted combat timeline. Camera 111 establishes for
1.25 seconds before the first beat begins.

The confirmed-death controller is the only camera progression authority:

- death 1 holds camera 111 for 0.85 seconds, selects camera 112, establishes it
  for 1.25 seconds, and then starts beat 2;
- death 2 holds camera 112 for 0.85 seconds, selects camera 113, establishes it
  for 1.25 seconds, and then starts beat 3;
- death 3 holds camera 113 for 0.85 seconds, selects camera 114, holds the final
  composition for 2.25 seconds, explicitly executes `k_oei_endconv` on the
  director at 3.10 seconds, and starts guarded `xt_post` at 3.35 seconds.

The builder's package readback rejects any dialogue with a second entry,
reply, link, wrong camera, wrong dwell, or timeline that can outlive the
dialogue. Focused source, package, and compiled-ABI tests cover the exact
contract. This is static and automated evidence only; the next retail run must
visibly confirm that each killing action remains under its intended camera and
that the final shot is readable before production dialogue opens.

### Location-native Miststep dependency and staged schema-11 candidate

The same retail run confirmed the custom Lightning and Drain beams but showed
no Miststep plume. The superseded `kxar_mstfx` was cloned from the
combat-impact model `v_rp_m_gen`; neither creature-root fields nor
`imp_impact_node` made that donor render through `ApplyEffectAtLocation`.

The current dependency snapshot replaces it with a token-safe green clone of
stock `v_grnpois_fnf`, the model used by the location-native
`VFX_FNF_GRENADE_POISON` contract. Numeric VFX label 9100 now has
`type_fd=F`, only `imp_root_m_node=kxar_mstfx`, and blank impact, other root,
and program fields. The six-emitter, roughly 3.97-second plume uses both
`fx_xmist.tga` and `fx_xmist1.tga`; both files are explicit external
dependencies and are hash-bound through candidate build, live validation, and
installation.

The schema-11 artifact built against prepared production session
`20260727T162457` and was then installed after the equivalent production
transaction `20260727T162702` committed. Teaser transaction
`20260727T164543458610` rehashed 206 live dependencies and installed:

- `xartease.mod`: 1,236,132 bytes,
  `DD43597065617A5CA312BB29CB8DB82D18EC83106EA59989F59776E45449D569`;
- structural proof: 94,169 bytes,
  `48550FAB8DF225ECCF5EE8F1AAA1999DEF622AF2806627C7BA6996E527928862`;
- teaser manifest: 90,932 bytes,
  `BD20FAB28001DF7581F19A00A177295C6896651E865C34D5B2A343E5A0930695`.

The committed staging manifest is 41,320 bytes with SHA-256
`8193FCE18CE5ABCAF50A08DEADD12EC5F71CE6237BD4B77FCCD3A5845FBFD6D4`.

This proves package structure and byte identity, not visible retail behavior.
The next capture must show both departure and arrival mist and must keep each
kill inside its assigned cinematic shot.

## 2026-07-27 verified-head and XariaBodyv2 dependency refresh

The teaser dependency contract was rebuilt from the uninstalled Patch Manager
candidate at
`logs/k2-plcaa-xaria-staging/20260727T215514/candidates`. The builder now
rejects the retired `p_xariah`/`p_xaria01` package and instead requires:

- `p_xariabb.mdl` SHA-256
  `3D1DEE052559B39D4B18CF4C8DBA110150E50B980FFADD4A23DE1518CDFA9EAC`;
- `p_xariabb.mdx` SHA-256
  `1E86FEA2F2CEA9B5F4D9F0DB1539D31156A7446BDF4E7EA68E452C4FC915ED01`;
- `p_xariab1.tga` SHA-256
  `80390E150E3F22790364E96EA77EAD6223E02D6830AB909E4953BAB85120E242`;
- verified `p_xariah6.mdl` SHA-256
  `35E8CDD7D3790F3E8171C82FEE650FBCC8E4FD7BD6EBB238C04E9A70CCA202E2`;
- verified `p_xariah6.mdx` SHA-256
  `B188292D04199C57824817150E106485BBEB9DB435385B4FEECC53B8750C2E6A`;
- authored `p_xaria06.tga` SHA-256
  `89399C4AA03B98D0548B3019ED1C5E58044CADA32FFCE8F24562CD6E21B49A72`;
- authored `p_xaria06.txi` SHA-256
  `93F2EE8298588CA48CF90F1E3B2C15AFFD58B9F6CE7421B2CF061A578B9287B0`;
- opaque final `po_pxaria.tga` SHA-256
  `DF0B6F5A1BD02EF441F6502A91844EE3127A8519DA2393A2F60F810CCB9A0C9D`;
- `heads.2da` row 199, referenced by `appearance.2da` row 725.

Two clean Ghost builds produced the same 55-resource module:

- `xartease.mod`: 1,236,132 bytes, SHA-256
  `DD43597065617A5CA312BB29CB8DB82D18EC83106EA59989F59776E45449D569`;
- `structural_proof.json`: 94,661 bytes, SHA-256
  `7F6047A416068F811A87D52E10823440A998229DDE9A7839F8EE8EBC8C51EB0E`;
- `xaria_teaser_manifest.json`: 91,398 bytes, SHA-256
  `47038EC3837BC4B94429E7D97452EA0122BC7B48681B995391D797C3C7F0ED6A`.

The MOD digest is identical to the already-installed schema-11 module. The
module does not embed Xaria's body or head; `XT_Xaria` resolves appearance row
725 at runtime. Therefore this refresh does not require replacing
`Modules/xartease.mod` when that exact digest is already installed. The next
game transaction must install the latest global Xaria Override/2DA candidate,
which supplies the verified head, XariaBodyv2-derived body, facial package,
production dialogue/voice assets, and custom-power dependencies. This remains
static/package evidence; a retail run must still visibly confirm the final
assembled character and the three cinematic power beats.

Focused verification:

- `tests/test_xaria_teaser_map.py`: 25 passed;
- `tests/test_xaria_teaser_map_stager.py`: 33 passed;
- `py tools/stage_k2_xaria_teaser_map.py`: passed with
  `installed=false`, `install_state=not_requested`, 108 external dependencies,
  and live dependency validation deferred until install.

## 2026-07-28 schema-12 once-only handoff and production dependency refresh

Schema 12 keeps the proven single-owner camera shell and adds an explicit
camera-114 handoff latch. `xt_dead` remains the only controller allowed to
advance confirmed deaths through cameras 112, 113, and 114. It sets the latch
only on the third confirmed death. `xt_post` refuses to open the production
conversation without that latch, and the interaction fallback can only request
the guarded handoff; it cannot bypass combat or synthesize completion. This
prevents the post-encounter dialogue from racing the final cinematic frame or
opening more than once.

The external dependency snapshot now carries the complete production package:

- nine locked DXD_003 story voice/LIP pairs;
- seven lesson voice/LIP pairs;
- five question-state action and condition script pairs;
- the seven lesson-taught base powers plus six level-up-only tier rows;
- spell rows 286 through 298, including numeric prerequisite chains and
  Dathomir Witch-only level 7/13 availability for the higher tiers.

The Ghost artifact was built twice from clean state with matching bytes:

- `xartease.mod`: 1,238,144 bytes, SHA-256
  `718072EA3F8B1C386B06A69A876025DC97FBDEB0BF6007BAF84AD46868652E32`;
- `structural_proof.json`: 101,065 bytes, SHA-256
  `6655002E9FF00E1CE3480C1FF425AED2363FD4F563F0EB03311EB57386681CBA`;
- `xaria_teaser_manifest.json`: 97,469 bytes, SHA-256
  `D5F3C6E8A1891EBB4D9837A46588C8E708175DB4760CC43F789ADC63319E466A`;
- `xartease.kmap`: 1,038,157 bytes, SHA-256
  `4D2847C1B17621A1B9F5A7BE9D0B450A6D1F134EE76A4FCC7C0E238AAB1FDA9E`;
- semantic resource digest:
  `2E2E8D820F22CBCE9F3B387ADC330E8B5090C867123AF72D256580E94BB0AD12`.

Binary readback contains camera IDs 110 through 115 and 55 module resources.
The complete Ghost teaser suite passed 26 tests; the paired Patch Manager
candidate suite passed 72 tests. Python compilation and Ruff checks passed.
This was an offline rebuild only. The next retail run must still visibly prove
the death-synchronized camera cuts and the custom-class level-up page.

## 2026-07-28 test-unlocked dependency rebuild and committed install

The production PLCaa package was intentionally staged with all lesson gates
unlocked for development. The teaser installer therefore rejected the older
normal-influence dependency snapshot at `cxar_l_veil.ncs`; this was the
correct fail-closed result and no partial module update occurred.

The Ghost builder was then run twice against the exact live test-unlocked
dependency set. Both clean builds matched, and the 26-test teaser suite,
Python compilation, and Ruff checks passed. Updated artifact evidence:

- `xartease.mod`: 1,238,144 bytes, SHA-256
  `718072EA3F8B1C386B06A69A876025DC97FBDEB0BF6007BAF84AD46868652E32`;
- `structural_proof.json`: 100,183 bytes, SHA-256
  `F359CB77838BF901E7B5A4B563C64C7E1482A97BD9031718D2C9CF1968D3A106`;
- `xaria_teaser_manifest.json`: 96,587 bytes, SHA-256
  `35A98188F64E72CFFD7D5BBCC7A28A7372D020AD991E1ADB8B58343D2A7AE59D`;
- all custom dependency evidence is labeled `live_game`.

Patch Manager then validated 249 live dependency records and committed the
module transaction. Its stage manifest is
`logs/k2-xaria-teaser-staging/20260728T023522901787/stage-manifest.json`
(SHA-256
`8A9A45958179231DF5808D3A564B51D30F8EC8B3C1F0CCEAFE344D6EB4D3636B`).
The installed module is byte-identical to the Ghost candidate. This records
package identity only; the assigned death-synchronized camera shots still
require visible retail confirmation.

## 2026-07-28 schema-13 natural three-camera handoff

The prior schema-12 test hung after Wraid 3 because combat camera 114 remained
inside the still-running private director dialogue. Calling
`k_oei_endconv` did not end that dialogue; it is an engine event wrapper, not
an active conversation-close operation.

Schema 13 removes camera 114 from the combat graph. The private DLG is now a
death-gated 111 -> 112 -> 113 graph with 400-millisecond polling links. Every
node has an advancing proof link followed by a negated child backlink, and
camera 113's proof link ends the graph naturally. `xt_camend`, installed as
the DLG `OnEnd` handler, validates the finished encounter, proof mask 7, all
three dead Wraids, and the once-only handoff latch before it calls `xt_post`.
The camera-114 marker remains available exclusively to the production
dialogue.

Camera 111 is byte-for-design unchanged. The revised cinematic views are:

- camera 112: position `(9.0, -13.0, 9.986575)`, target
  `(5.5, -20.0, 11.0)`, FOV 58;
- camera 113: position `(-2.0, -10.0, 10.048286)`, target
  `(4.5, -17.5, 11.0)`, FOV 58.

Each beat clears the target's actions, explicitly plays the stock Wraid
`cspasm` reaction for 0.65 seconds, and then applies lethal universal damage
so the stock `cdie` animation is available before the graph advances.

Two clean builds matched. Current artifact evidence:

- `xartease.mod`: 1,246,279 bytes, SHA-256
  `277AC9611BCCF3CDADEBE55566F96E177BBD42DABA7A9107F3086257C88DADCD`;
- `structural_proof.json`: 101,665 bytes, SHA-256
  `0C03E2C167C34FF36D9050356BE2BB478D52D8E51871342704F85E4C5A7AC778`;
- `xaria_teaser_manifest.json`: 98,830 bytes, SHA-256
  `32920C3D54690D9C4F60BE5CE027FD6DF1B84916FFFDD4F56EEFDC0091CBE46A`;
- 59 packed module resources, including 26 private/director resources;
- 27 focused Ghost tests passed; the paired Patch Manager suite passed 66.

The paired installer now independently rejects non-executable private NCS
payloads, missing `xartease.git` encounter placements, and private Xaria UTCs
that lose `xt_click`, `xt_hb`, or `ScriptEndDialogu=xt_enddlg`.

Patch Manager installed the byte-identical module transactionally under
`logs/k2-xaria-teaser-staging/20260728T095035056770/stage-manifest.json`
(SHA-256
`5FBF86A506D6879F67FF4ADDB2819B5ACBC3E7CBF248BE57779741C01DD44C74`).
This remains structural evidence until the user confirms framing, reactions,
camera release, and the one-time recruitment dialogue in retail.

## 2026-07-28 schema-14 retail DLG-delay unit correction

The first schema-13 retail run killed Wraid 1 but then appeared to freeze on
camera 111. The death proof and beat script were not the failure. The invalid
assumption was that a DLG node's integer `Delay` field is measured in
milliseconds. Both installed modules contained `[400, 400, 400]`, so retail
held each silent camera node for approximately 400 seconds before it could
reevaluate the death-proof link.

The unit is now independently proven as seconds:

- retail `FUN_006C67E0` reads `Delay`, `FUN_006CC140` passes it to
  `FUN_006C9770`, and `FUN_00545470` converts the duration using the integer
  `1000` at `0x00992384`;
- a scan of 1,242 stock KOTOR II dialogues found no `Delay=400`; stock
  cinematic nodes use small whole-second values such as 1, 3, 4, 5, and 13;
- KotOR.js multiplies the field by 1000, while reone treats it as a
  seconds-based timer.

Schema 14 therefore uses the smallest safe integer poll, `[1, 1, 1]`. Zero
was deliberately rejected because the graph contains child backlinks and a
zero-time cycle could spin without yielding. The builder, package validator,
and focused tests now require the exact one-second values. Ghost Studio's PIE
timing helper was corrected at the same time: its public argument is
`delay_seconds`, it no longer divides by 1000, and the generated Core Scene
and Core Tools payloads were regenerated.

Verification:

- 36 focused teaser/PIE tests passed;
- 22 embedded-Python payload tests passed;
- two clean teaser builds matched;
- candidate `xartease.mod`: 1,246,279 bytes, SHA-256
  `95123423CB3AAD35BF5754B499B95125F58459FFA72AE0EE45FA50C52459259B`;
- `structural_proof.json`: 101,652 bytes, SHA-256
  `49FBA487564F0FD797708092046A9E1B5D876C74BD096B377CF744A995556398`;
- `xaria_teaser_manifest.json`: 98,830 bytes, SHA-256
  `DC9E6EBCD343A280E4FBC2DF7264C703C701D277A373193D5D2C36D32BAA1552`.

Patch Manager installed that exact candidate transactionally under
`logs/k2-xaria-teaser-staging/20260728T104139571289/stage-manifest.json`.
This proves package identity and graph structure only. Retail acceptance still
requires the user to see camera 111 advance to 112 after Wraid 1, then to 113,
and finally release into the recruitment dialogue.

## 2026-07-28 acknowledged post-combat conversation retry

The next retail run visibly completed all three combat beats and naturally
released camera 113, but the production recruitment dialogue did not open.
The invalid assumption was that scheduling `ActionStartConversation` and
setting local Boolean 61 first proved the conversation had started. Retail can
silently reject that queued action during camera-dialogue and combat teardown,
after which the optimistic latch prevented every heartbeat retry.

The shared production dialogue now supplies the acknowledgement:
`xaria.dlg` runs external dependency `kxar_dlgack.ncs` on its first entry and
sets Boolean 61 only when that node actually executes. `xt_post` now uses the
stock combat-to-dialogue teardown pattern on Xaria and the PC, waits 1.00
second before requesting the conversation, and retries after 2.00 seconds
while acknowledgement is absent. `xt_hb` retains the state-2 recovery path,
and the private camera DLG routes both `OnEnd` and `OnAbort` to `xt_camend`.

Verification:

- 27 focused Ghost teaser tests passed;
- the paired Patch Manager Xaria suite passed 73 tests;
- two clean builds matched and structural proof reports `PASS`;
- `xartease.mod`: 1,246,513 bytes, SHA-256
  `300EA47A14A2C2215AEA68200EEA8870B015D5FC99EA509A2077B024EF553276`;
- `structural_proof.json`: 103,020 bytes, SHA-256
  `106348C6BCD429D31F3C43E10F202A685554CD86BC485F05F51CFAE484DC3F43`;
- `xaria_teaser_manifest.json`: 100,133 bytes, SHA-256
  `35176A3FFB97DF45BFA5E2CB5B9646F23D6BF88FBEB236C0D65C7D919B0CBFB2`;
- Patch Manager installed the byte-identical module transactionally under
  `logs/k2-xaria-teaser-staging/20260728T183453232977/stage-manifest.json`
  after validating 259 live dependencies.

The package readback proves `OnEnd=OnAbort=xt_camend` and the acknowledgement
dependency matches the installed production candidate. It does not prove
retail behavior. The remaining acceptance gate is one production recruitment
dialogue immediately after Wraid 3, with no encounter restart.

## 2026-07-28 camera-OnEnd handoff deadlock and independent recovery

The next retail run again completed all three Wraid kills and the complete
three-camera combat sequence, but no production dialogue began. Interacting
with Xaria afterward also did nothing. This invalidated a second assumption:
even with a naturally terminating DLG, `OnEnd`/`OnAbort` cannot be the sole
authority that unlocks postcombat gameplay.

The deadlock was deterministic:

- `xt_dead` committed private state 2 and proof mask 7 but did not schedule
  production dialogue;
- only `xt_camend` could set camera-handoff Boolean 60;
- `xt_post` required Boolean 60 before doing any work;
- heartbeat called the same gated `xt_post`, so it could never recover; and
- `xt_click` accepted only production globals 2 or 4, while the missed callback
  left the global at 0 or 1.

The last retail-proven schema that visibly reached postcombat subtitles did
not rely on the callback. Schema 14 now restores that architecture without
reverting the working three-camera combat:

1. proof mask 7 schedules `xt_post` independently after 0.90 seconds;
2. `xt_post` no longer requires Boolean 60;
3. it polls the real `XT_Director` with `GetIsInConversation` every 0.50
   seconds and lets the camera DLG end naturally;
4. after release it records Boolean 60 as telemetry, performs the guarded
   combat teardown, and requests `xaria.dlg` after the retail-proven 0.35
   seconds;
5. Boolean 61 remains post-launch evidence from `kxar_dlgack`, and heartbeat
   retries until that acknowledgement; and
6. a click in completed private state retries the handler even when the
   production global is still 0 or 1.

Do not reintroduce `k_oei_endconv` as a close operation. Retail already proved
that it is an event wrapper and does not end the active camera conversation.

Verification and staging:

- all 27 focused Ghost teaser tests passed, including native K2 ACTION operand
  validation and binary DLG readback;
- two clean candidate builds matched;
- `xartease.mod`: 1,246,513 bytes, SHA-256
  `D74DD904C9CFA2321503C8E4A6A3B89487A4BC40142CEEDDBB481AAA5BB4CDC2`;
- `xt_dead.ncs` SHA-256
  `57A041BD9A99F5A620574799DC0B525CD4C292FBCAFF507F6E62854942CF769`;
- `xt_post.ncs` SHA-256
  `AA009975D54430D4CA35F8E5B7638E4DEBB96AD30788BAF624766708854DB42D`;
- `xt_hb.ncs` SHA-256
  `60DD5A965CD7AD47764CE2AAF3C069A53CC613C9095F83DCE01A6179FCEAD289`;
- installed transaction:
  `logs/k2-xaria-teaser-staging/20260728T201304618797/stage-manifest.json`.

The installer validated the newly staged PLCaa dependency manifest before
committing the byte-identical xartease module. This is structural and package
identity evidence only. The required retail gate remains: all three kills and
cameras complete, the recruitment dialogue begins once without interaction,
and clicking Xaria also recovers it if the automatic launch is deliberately
missed.

## 2026-08-03 — gender-aware reaction LIP dependency refresh

The teaser's external dependency contract now matches the production
encounter's complete 49-line voice set. The obsolete single
`cxar_atton.ncs` dependency was replaced by `cxar_atton_m.ncs` and
`cxar_atton_f.ncs`; `997atton002` was added for the female-player reaction.
Atton's two LIPs and Kreia's LIP were regenerated from reviewed
faster-whisper large-v3 word timestamps rather than uniform whole-line
grapheme timing.

Pinned LIP SHA-256 values:

- `997atton001.lip`:
  `DDDDE6BF03600C455A9AAD5D2E3344A9806C76BB8B1F927DA357CCC839D8E0EF`;
- `997atton002.lip`:
  `A8026DCAC4E5B9F4EF2C38D0F1FFE422468217A7BE0A0C171AD7446BB65FB373`;
- `997kreia001.lip`:
  `32F8274B6205810E696C284006BD9325E324B2CC73596C9FD540F8A15CF81190`.

All 30 focused teaser tests passed. Two clean teaser builds completed, and
KPM transactionally installed the 1,247,892-byte module with SHA-256
`EA6F0E1AB59CFC27811C14003050B02921F19E43EA7302101A9E6C8AAC7088A6`
under
`logs/k2-xaria-teaser-staging/20260803T170146849133/stage-manifest.json`.
The install validated 365 live dependencies. This is structural, alignment,
package, and installed-byte evidence only; visible in-game facial sync remains
pending.

## 2026-08-04 — reaction LIPs advanced to audio-derived mouth cues

Retail testing invalidated the preceding word-aligned LIPs: Atton's added
reaction still did not animate his lips correctly. Kreia was not yet tested,
but shared the same generator and was corrected in the same pass. The failed
method aligned words to audio and then divided each word evenly by transcript
characters, choosing visemes from English spelling. It was not a reliable
phonetic model.

KPM now builds all three reaction LIPs from reviewed Rhubarb Lip Sync 1.13.0
PocketSphinx mouth cues bound to the exact source-audio hashes. Ghost Studio's
dependency pins are:

- `997atton001.lip`:
  `A4D67480F3424C4FAC58590C7001F74AFE76117BE1E847516B7D7015CC6A158B`;
- `997atton002.lip`:
  `F2F4AA61CF32012C5F084B087EA8DA68F7ADA83069829D94155ABA4DFBD5AFE3`;
- `997kreia001.lip`:
  `15FA2FDF1D812002107CB15C294F02C37EF1BE1366F51E8A44C30FCA52EF5FD9`.

The rebuilt teaser retains the byte-identical 1,247,892-byte
`xartease.mod`, SHA-256
`EA6F0E1AB59CFC27811C14003050B02921F19E43EA7302101A9E6C8AAC7088A6`,
because these reaction LIPs are loose dependencies. Updated artifact hashes:

- `structural_proof.json`:
  `600B8E5317036A9C7D03403CB0645F2264BDB642931563B2D0E2743D6A167470`;
- `xaria_teaser_manifest.json`:
  `7E01B6F5D7DBB4F0F94ABE83827EC296EE0C736F6C13BF35ADD8DF3A17EEAA17`.

All 30 focused Ghost Studio teaser tests and 75 affected KPM tests pass. KPM
transactionally installed the updated dependency set under
`logs/k2-xaria-teaser-staging/20260804T162120563070/stage-manifest.json`
(SHA-256
`DAC2E204FC5555A7CE265A241F50738B5D7A4A8237CD6A83A9D9EFBD0CD6138F`).
Installed loose LIP hashes exactly match the pins above. Visible in-game sync
remains `candidate_unverified` until Atton and Kreia are retested.

## 2026-08-06 — accepted production Nydak encounter replaces the private clone

The teaser builder now emits the byte-verified production PLCaa encounter
spine instead of packing the older private `xt_*` candidate. Its source of
truth is KPM staging session
`logs/k2-plcaa-xaria-staging/20260806T152132/candidates/Override`. The module
contains `xar_enc.utc`, `xar_nydak1.utc` through `xar_nydak3.utc`,
`xar_introtrig.utt`, `xar_director.utp`, `xar_clock.utp`, `xar_cam.dlg`,
`xaria_blade.uti`, the three `kxar_*` sound templates, and the complete set of
24 production `kxar_*` NCS files. No `xt_*` resource is packed.

The placement contract is also production-identical: actor tag
`Xaria_Encounter`; Nydak tags `Xaria_Wraid_1` through `Xaria_Wraid_3`; trigger
tag `Xaria_Intro_Trigger`; director and timeline tags `Xaria_Director` and
`Xaria_Timeline`; and sound tags matching their `kxar_*snd` resrefs. The camera
dialogue uses cameras 111, 112, and 113 with 3/7/8-second pacing, begins through
`kxar_cam1`, and exits through `kxar_camend`. Cameras 110 through 115 remain in
the module for compatibility.

Two clean builds were byte-identical. The resulting 69-resource
`artifacts/xaria_teaser/xartease.mod` is 1,252,606 bytes with SHA-256
`74549DCD17D7A53C97E85A7DA3C1F6D391A786D079E712DC556281E3B1C3AA59`.
`structural_proof.json` has SHA-256
`890318650E0FA90F62F573BC3329DA235139ECE3C25DC9BCAF6314B66B6325CE`;
`xaria_teaser_manifest.json` has SHA-256
`1C66E21757F432043CE90C9AC5922C61A167FAB4A121BA80C5D87F2EAE2DA36A`.
All 31 focused builder tests and all 12 KPM installer tests pass; Python compile
and Ruff checks also pass.

KPM transactionally installed the candidate while KOTOR II was closed under
`logs/k2-xaria-teaser-staging/20260806T182944867038/stage-manifest.json`.
All 381 external dependencies validated, and the live `Modules/xartease.mod`
hash matches the build. The previous module is preserved in that transaction
as `Modules/xartease.mod.before` with SHA-256
`EA6F0E1AB59CFC27811C14003050B02921F19E43EA7302101A9E6C8AAC7088A6`.
Runtime status remains `candidate_unverified` until the complete Nydak
cinematic, dialogue handoff, and recruitment are visibly accepted in game.
