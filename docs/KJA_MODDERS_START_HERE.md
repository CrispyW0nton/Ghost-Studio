# KJA modders: start here

Kotor Academy: Expanded Galaxy (KJA) runs both KOTOR 1 and KOTOR 2 content on
the KOTOR 2 engine. Ghost-Studio is the recommended tool for making KJA mods.
This page covers what you need to know before your first one.

## 1. Install Ghost-Studio

Follow the [installation steps in the README](../README.md#installation). When a
prebuilt download is available it is listed on the
[Releases page](https://github.com/CrispyW0nton/Ghost-Studio/releases).

## 2. Point it at the game

Open **Settings → Game Paths** and set the KOTOR II folder that KJA is installed
into. Ghost-Studio reads the game's own archives; it never overwrites the
installation unless you choose an explicit export or install action.

## 3. Pick the right studio

| You want to... | Use |
| --- | --- |
| Browse and open existing models, textures and modules | **Content Browser** |
| Make or edit a character, head, outfit or creature | **Character Studio** |
| Move animations from one skeleton to another | **Retarget Studio** |
| Change an existing module (placeables, creatures, doors, triggers, cameras) | **Module Studio** |
| Build a new area or planet (Blender-first import, walkmesh, lightmaps) | **Map Studio** |
| Write or fix scripts, dialogue and 2DA tables | **Scripting Suite** |

## 4. KJA rules that save you a bug report

- **Target KOTOR 2.** KOTOR 1 content in KJA runs on the KOTOR 2 engine, so build
  for KOTOR 2. Ghost's K1 → K2 model porter helps bring KOTOR 1 assets across.
- **New modules start in Classic mode.** JKA mode needs collision that is
  qualified for each module. Tools to do that yourself are planned after the
  beta. Until then your module plays in Classic mode.
- **Never break another mod.** Add rows to shared 2DA tables instead of editing
  existing ones (Ghost's append-only 2DA writing keeps the original rows
  untouched). Don't replace files another mod ships.
- **Ship area changes inside `Modules/<module>.mod`.** An area's `.git` file
  placed loose in `Override` takes priority over save games, so encounters can
  replay.
- **Compile scripts in Ghost-Studio.** It repairs and verifies the PyKotor
  compiler before every compile and refuses bytecode the game would abort on.
- **Keep mesh nodes at or under 32,767 vertices.** The engine stores per-node
  vertex counts as signed 16-bit values. Ghost refuses larger nodes on export.
- **Creature models need the engine's dummy nodes:** `cutscenedummy` →
  `rootdummy`, `camerahook`, `impact` and `talkdummy`. Ghost's module validator
  warns when a standalone creature model is missing them.
- **Test in the game.** The editor preview is not the game. Look at your mod in
  KJA before you release it.

## 5. Get help

Ask in `#ghost-studio-help` on the KJA Discord, share your work in
`#ghost-studio-showcase`, and post feature requests or bugs in
`#ghost-studio-requests` or as a
[GitHub issue](https://github.com/CrispyW0nton/Ghost-Studio/issues).
