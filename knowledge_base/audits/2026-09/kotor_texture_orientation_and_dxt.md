# KOTOR texture orientation and DXT authoring: what the DXM_101 in-game test proved

Date: 2026-09-28. Source: the Dathomir DXM_101 module (an Unreal master converted through Ghost's authored-module
export) tested in KOTOR II (Steam, TSLRCM 1.8.6), decode tests against the installed game data.

## 1. PyKotor's DXT encoder is not usable for authoring

`TPC.set_single(..., TPCTextureFormat.RGB, ...)` followed by `TPC.convert(TPCTextureFormat.DXT1)` produced, for a
2048 x 2048 rock/moss albedo page:

- base level: colour-shifted and blocky (mean RGB 34/45/14 for an 84/66/41 source, mean abs error 32/255);
- every smaller mip: solid black.

In game this read as green, pixelated textures near the camera and black surfaces at distance (engine picks lower
mips). The header fields were vanilla-correct (data_size, encoding 2, mip count), so nothing flagged it.

Fix: `src/core/graphics/tpc_dxt_writer.py` (`encode_tpc_dxt`): Pillow BC1/BC3 per mip level, box-filtered chain to
1 x 1, vanilla header, optional TXI trailer. Round-trip error ~6/255 at every level.

Rule: after writing any compressed TPC, decode every mip level (PyKotor `read_tpc` + `convert(RGB)`) and compare with
the source resized to that level. A header check is not enough.

## 2. UV / row orientation

- A raw decode of a vanilla TPC (e.g. `load_101per` from `swpc_tex_gui.erf`) reads upright: TPC rows are stored top to
  bottom.
- The engine samples the first stored row at MDX UV v = 0 (top-left origin in MDX storage).
- Ghost's glTF and Blender-FBX importers already mark their nodes `uv_v_flip=False`, which makes
  `MDLBinaryWriter._uv_pair_for_mdx` write v -> 1 - v; native KOTOR models keep their rows.
- Authored `ImportedMeshSurface` data is written as given (no node flag). A caller whose UVs have a bottom-left origin
  (v = 1 at the image's top row, e.g. an atlas rasterised with texel y = v * height and saved with row 0 = top) must
  flip v itself. DXM_101 does this for albedo UVs, lightmap UVs, plant cards, trees and water.
- Ghost's five-face sky panels use v = 0 at the panel bottom: store their images bottom row first.

Open: the Map Studio selected-surface lightmap transaction rasterises with v = 1 - (y + 0.5) / res and writes rows in
that order; its in-game orientation is still unproven - check it against this rule before the first game proof.

## 3. Culled geometry needs a closed backing

Cutting faces that no walkable viewpoint sees, then decimating, opened small gaps between cladding pieces that the sky
showed through. Keep the base terrain (canyon shells) closed behind any cladding and let decimation thin it instead.
