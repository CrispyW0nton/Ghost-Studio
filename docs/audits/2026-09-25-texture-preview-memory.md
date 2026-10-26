# Texture preview memory investigation

Owner: LordVaderCW
Date: 2026-09-25
Branch: codex/asset-memory
Baseline: e17555717568179968ae5157bf3535f3dd8949ac

## Result and limits

The software renderer's derived mip cache retained old pixels across texture
source changes. It also keyed pixels by `id(image)` without validating source
identity, allowing address reuse to return a different texture's pixels.

The fix caps this disposable cache at 128 entries and 32 MiB of pixel storage.
It does not cap total process memory, base textures, source bytes, or GPU memory.
It does not change module assembly, model parsing, NPC counts, or VFX simulation.
The reported whole-PC crash has not been reproduced or attributed.

Rendering owns this policy. This checkout resolves the package-local Python
implementation from Core.Rendering; no matching root source exists. The GUI
texture-cache file is a compatibility facade.

## Evidence

A bounded synthetic workload loaded 160 distinct 512 x 512 RGBA images through
`TextureCache.get`, then requested half-resolution previews through `get_mip1`.
Before the fix the base pixels occupied 160 MiB, and all 160 derived images
(40 MiB) remained after `set_game_library` cleared the base cache.

After the fix, the same workload retains 128 derived images (32 MiB). Switching
the library releases all derived entries. Eight new regression cases failed
against the baseline for retention, missing limits, and wrong-source reuse;
they passed after the correction. Additional tests cover paint races, source
collection, authored pixels, and no repeated resizing when the cache is full.

An initial eviction-based implementation was rejected: cycling through 160 live
textures repeatedly resized them and cost approximately 633 ms in the local
probe. Stable admission with original-image fallback reduced that repeated lookup
probe to under 1 ms. This is a cache microbenchmark, not an application FPS claim.

The game-file MCP comparison matched for K2 `PLC_bench` and `001ebo1` (3 and 60
nodes respectively). Those tools validate the original workspace pipeline, not
the isolated branch's UI. The isolated checkout's real K2 texture parity tests
verify that its decoder pixels and metadata remain unchanged.

## Verification

Executed in the isolated checkout with Python 3.14:

```text
python -m pytest tests/test_tpc_viewport_mip_fast_path.py tests/test_resource_manager_revision_concurrency.py tests/test_pygfx_renderer_backend.py::test_texture_cache_patches_clipped_region_without_flipping_or_replacing_image tests/test_native_python_payloads.py::test_python_payload_copies_are_byte_identical_and_manifested -q
24 passed
```

Regenerated the Rendering payload and built the native host in Debug|x64. The
host build regenerated a previously stale Tools manifest hash for the existing
placeable-builder controller. This generated metadata repair is included so
payload verification is valid on the committed state.

Launched that Debug executable from Visual Studio using its debugger. Startup
reported the ModernGL backend and the main window became available to Windows
accessibility inspection. Windows screenshot capture repeatedly failed with
`window capture timed out: timed out waiting on channel`; Visual Studio capture
also failed. No visible room-loading, camera-motion, or renderer-parity pass is
claimed. Visible qualification is outstanding.

## Review

The local qwen30lead model reviewed the bounded design (669 reported tokens).
Its recommendations were verified directly. It was no longer loaded at the
final code-review check, so no final independent local diff review ran and no
cloud reviewer was substituted. Manual review covered identity checks, lock
scope, invalidation during resize/painting, dead-source cleanup, fallback use
by renderer callers, and authored-pixel preservation. Simplification used a
plain dict and kept cache hits ahead of size calculation and capacity scans.

Release status: not ready for a claim that the reported crashes are fixed;
native visible qualification and reporter crash details are still needed.
