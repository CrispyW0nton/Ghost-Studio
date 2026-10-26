"""KWORLD2 sidecar writers and readers: static-area geometry and textures for the KOTOR II Diligent area renderer.

Added 2026-09-29 for the DXM_101 (Dathomir) area renderer, worked jointly with GPT-6 Astra; the byte layouts follow
Astra's ``kworld_sidecar_spec_v001.md`` (RendererPolycountK2/astra) exactly, plus Claude's amendment 2 (per-vertex
baked light in the reserved vertex u32, chunk flag bit 0).

KWCHUNK2 v1 (little-endian, 128-byte header)::

    0   byte[8]  "KWCHUNK2"          8,12  u32 version=1, header_bytes=128     16,20 u32 flags, vertex_stride=64
    24..36 u32   vertex_count, index_count, submesh_count, lod_id
    40..64 u64   file_bytes, vertex_offset, index_offset, submesh_offset      72,84 float3 aabb min, max
    96  byte[32] sha256 of bytes [128, file_bytes)

Vertex (64 bytes): position f32x3 @0, normal f32x3 @12, tangent f32x4 @24 (w = +/-1), uv0 f32x2 @40, uv1 f32x2 @48,
colour RGBA8 @56, reserved u32 @60 (flag bit 0 set: packed RGB10A2 baked light). Indices u32, triangle list.
Submesh (32 bytes): u32 material_id, first_index, index_count, flags, 4 x u32 zero. Payload offsets 16-byte aligned.

KWTEX2 v1 (128-byte header)::

    0   byte[8]  "KWTEX2\\0\\0"      8,12  u32 version=1, header_bytes=128
    16..28 u32   format (1 BC1, 2 BC3, 3 BC5, 4 BC7), colour_space (0 linear, 1 sRGB), width, height
    32,36 u32    mip_count, flags     40..56 u64 file_bytes, mip_table_offset, payload_offset
    64  byte[32] sha256 of bytes [128, file_bytes)      96 byte[32] reserved

Mip record (64 bytes): u32 width, height; u64 offset, byte_count, reserved; byte[32] sha256 of the mip bytes.
Rows are stored bottom-up (OpenGL origin), so callers pass images in their natural top-down orientation and the
writer flips once. BC1/BC3 come from Pillow's DDS encoder, BC5 from the numpy BC4 encoder here, BC7 from a mode-6 encoder here.

Headless and file-based; numpy arrays in, files out; ``read_*`` functions exist for round-trip tests.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

KWCHUNK2_MAGIC = b"KWCHUNK2"
KWTEX2_MAGIC = b"KWTEX2\0\0"
VERTEX_STRIDE = 64
HEADER_BYTES = 128
CHUNK_FLAG_VERTEX_LIGHT = 1
TEX_FORMATS = {"BC1": 1, "BC3": 2, "BC5": 3, "BC7": 4}
BLOCK_BYTES = {1: 8, 2: 16, 3: 16, 4: 16}
CHUNK_CAPS = {"vertices": 1_000_000, "indices": 3_000_000, "submeshes": 4096, "payload_bytes": 32 << 20}


def _align16(n: int) -> int:
    return (n + 15) & ~15


def pack_rgb10a2(rgb: np.ndarray) -> np.ndarray:
    """Float RGB in [0, 1] (N, 3) to packed u32 with 10 bits per channel and alpha bits 3."""

    q = np.clip(np.rint(np.asarray(rgb, np.float64) * 1023.0), 0, 1023).astype(np.uint32)
    return q[:, 0] | (q[:, 1] << 10) | (q[:, 2] << 20) | (np.uint32(3) << 30)


def unpack_rgb10a2(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, np.uint32)
    return np.stack([(p & 1023), (p >> 10) & 1023, (p >> 20) & 1023], 1).astype(np.float64) / 1023.0


def write_kwchunk2(path: str | Path, *, positions: np.ndarray, normals: np.ndarray, indices: np.ndarray,
                   submeshes: Sequence[tuple[int, int, int, int]], tangents: np.ndarray | None = None,
                   uv0: np.ndarray | None = None, uv1: np.ndarray | None = None, colors_rgba8: np.ndarray | None = None,
                   light_rgb: np.ndarray | None = None, lod_id: int = 0) -> dict[str, Any]:
    """Write one geometry chunk; returns its record (counts, aabb, sha256, bytes) for the manifest."""

    P = np.ascontiguousarray(positions, np.float32); n = len(P)
    N = np.ascontiguousarray(normals, np.float32)
    idx = np.ascontiguousarray(indices, np.uint32).reshape(-1)
    if n == 0 or len(idx) == 0 or len(idx) % 3:
        raise ValueError("a chunk needs vertices and a triangle list")
    if n > CHUNK_CAPS["vertices"] or len(idx) > CHUNK_CAPS["indices"] or len(submeshes) > CHUNK_CAPS["submeshes"]:
        raise ValueError("chunk exceeds the KWCHUNK2 caps; split it")
    if int(idx.max()) >= n:
        raise ValueError("index out of range")
    if not np.isfinite(P).all() or not np.isfinite(N).all():
        raise ValueError("non-finite positions or normals")
    T = np.ascontiguousarray(tangents, np.float32) if tangents is not None else np.tile(np.array([1, 0, 0, 1], np.float32), (n, 1))
    U0 = np.ascontiguousarray(uv0, np.float32) if uv0 is not None else np.zeros((n, 2), np.float32)
    U1 = np.ascontiguousarray(uv1, np.float32) if uv1 is not None else np.zeros((n, 2), np.float32)
    C = np.ascontiguousarray(colors_rgba8, np.uint8) if colors_rgba8 is not None else np.full((n, 4), 255, np.uint8)
    flags = 0
    R = np.zeros(n, np.uint32)
    if light_rgb is not None:
        R = pack_rgb10a2(light_rgb); flags |= CHUNK_FLAG_VERTEX_LIGHT
    vb = np.zeros(n, dtype=np.dtype([("p", np.float32, 3), ("n", np.float32, 3), ("t", np.float32, 4), ("uv0", np.float32, 2),
                                     ("uv1", np.float32, 2), ("c", np.uint8, 4), ("r", np.uint32)]))
    vb["p"] = P; vb["n"] = N; vb["t"] = T; vb["uv0"] = U0; vb["uv1"] = U1; vb["c"] = C; vb["r"] = R
    assert vb.dtype.itemsize == VERTEX_STRIDE
    sub = sorted(submeshes, key=lambda s: s[1])
    covered = 0
    for mat, first, count, sflags in sub:
        if first != covered or count <= 0 or count % 3:
            raise ValueError("submesh ranges must be sorted, contiguous and whole triangles")
        covered += count
    if covered != len(idx):
        raise ValueError("submeshes must cover every index")
    sb = np.zeros((len(sub), 8), np.uint32)
    for i, (mat, first, count, sflags) in enumerate(sub):
        sb[i, :4] = (mat, first, count, sflags)
    v_off = HEADER_BYTES; i_off = _align16(v_off + n * VERTEX_STRIDE); s_off = _align16(i_off + len(idx) * 4)
    file_bytes = s_off + len(sub) * 32
    if file_bytes - HEADER_BYTES > CHUNK_CAPS["payload_bytes"]:
        raise ValueError("chunk payload over 32 MiB; split it")
    body = bytearray(file_bytes - HEADER_BYTES)
    body[v_off - HEADER_BYTES:v_off - HEADER_BYTES + n * VERTEX_STRIDE] = vb.tobytes()
    body[i_off - HEADER_BYTES:i_off - HEADER_BYTES + len(idx) * 4] = idx.tobytes()
    body[s_off - HEADER_BYTES:s_off - HEADER_BYTES + len(sub) * 32] = sb.tobytes()
    digest = hashlib.sha256(body).digest()
    lo = P.min(0); hi = P.max(0)
    head = struct.pack("<8sIIIIIIIIQQQQ3f3f", KWCHUNK2_MAGIC, 1, HEADER_BYTES, flags, VERTEX_STRIDE, n, len(idx), len(sub), lod_id,
                       file_bytes, v_off, i_off, s_off, *lo.tolist(), *hi.tolist())
    head += digest
    head += b"\0" * (HEADER_BYTES - len(head))
    data = head + bytes(body)
    Path(path).write_bytes(data)
    return {"path": str(path), "bytes": file_bytes, "sha256": hashlib.sha256(data).hexdigest(), "vertices": n,
            "triangles": len(idx) // 3, "submeshes": len(sub), "lod_id": lod_id, "flags": flags,
            "aabb_min": lo.round(4).tolist(), "aabb_max": hi.round(4).tolist()}


def read_kwchunk2(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    magic, ver, hb, flags, stride, n, ni, ns, lod, fb, v_off, i_off, s_off = struct.unpack_from("<8sIIIIIIIIQQQQ", data, 0)
    if magic != KWCHUNK2_MAGIC or ver != 1 or hb != HEADER_BYTES or stride != VERTEX_STRIDE or fb != len(data):
        raise ValueError("not a KWCHUNK2 v1 file")
    if hashlib.sha256(data[HEADER_BYTES:fb]).digest() != data[96:128]:
        raise ValueError("payload hash mismatch")
    vb = np.frombuffer(data, np.dtype([("p", np.float32, 3), ("n", np.float32, 3), ("t", np.float32, 4), ("uv0", np.float32, 2),
                                       ("uv1", np.float32, 2), ("c", np.uint8, 4), ("r", np.uint32)]), n, v_off)
    idx = np.frombuffer(data, np.uint32, ni, i_off)
    sub = np.frombuffer(data, np.uint32, ns * 8, s_off).reshape(ns, 8)
    if ni and int(idx.max()) >= n:
        raise ValueError("index out of range")
    return {"flags": flags, "lod_id": lod, "positions": vb["p"], "normals": vb["n"], "tangents": vb["t"], "uv0": vb["uv0"],
            "uv1": vb["uv1"], "colors": vb["c"], "reserved": vb["r"], "indices": idx, "submeshes": sub[:, :4],
            "aabb": struct.unpack_from("<3f3f", data, 72)}


# ---------------------------------------------------------------- textures

def _bc4_blocks(channel: np.ndarray) -> bytes:
    """BC4 (single channel, unsigned) for a (H, W) uint8 image whose sides are multiples of 4; 8 bytes per block."""

    h, w = channel.shape
    blk = channel.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 16).astype(np.int32)
    mx = blk.max(1); mn = blk.min(1)
    same = mx == mn
    mx = np.where(same, np.minimum(mx + 1, 255), mx); mn = np.where(same & (mx == mn), np.maximum(mn - 1, 0), mn)
    # 8-value palette (a0 > a1 form): a0, a1, then 6 interpolants
    a0 = mx.astype(np.float64); a1 = mn.astype(np.float64)
    pal = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) / 7 for k in range(1, 7)], 1)          # (B, 8)
    ind = np.abs(blk[:, :, None] - pal[:, None, :]).argmin(2).astype(np.uint64)                   # (B, 16)
    bits = np.zeros(len(blk), np.uint64)
    for k in range(16):
        bits |= ind[:, k] << np.uint64(3 * k)
    out = np.zeros((len(blk), 8), np.uint8)
    out[:, 0] = mx; out[:, 1] = mn
    b = bits.astype("<u8").view(np.uint8).reshape(-1, 8)
    out[:, 2:8] = b[:, :6]
    return out.tobytes()


_BC7_W4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], np.int64)


def _bc7_mode6_blocks(rgba: np.ndarray) -> bytes:
    """BC7 mode 6 (one subset, RGBA 7.7.7.7 endpoints + p-bits, 4-bit indices) for an (H, W, 4) uint8 image whose sides
    are multiples of 4. Endpoints are the per-channel block extremes; good for masks, data and smooth colour."""

    h, w = rgba.shape[:2]
    blk = rgba.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.int64)
    lo = blk.min(1); hi = blk.max(1)                              # (B, 4)
    e0 = (lo >> 1); p0 = (lo & 1).sum(1) >= 2                     # p-bit: majority of the low bits
    e1 = (hi >> 1); p1 = (hi & 1).sum(1) >= 2
    c0 = (e0 << 1) | p0[:, None]; c1 = (e1 << 1) | p1[:, None]     # reconstructed 8-bit endpoints
    pal = (c0[:, None, :] * (64 - _BC7_W4)[None, :, None] + c1[:, None, :] * _BC7_W4[None, :, None] + 32) >> 6   # (B, 16 weights, 4)
    err = ((blk[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(3)  # (B, 16 texels, 16 weights)
    idx = err.argmin(2)                                            # (B, 16)
    swap = idx[:, 0] >= 8                                          # the first index has no MSB: swap endpoints
    idx = np.where(swap[:, None], 15 - idx, idx)
    e0s = np.where(swap[:, None], e1, e0); e1s = np.where(swap[:, None], e0, e1)
    p0s = np.where(swap, p1, p0); p1s = np.where(swap, p0, p1)
    lo64 = (np.uint64(1) << np.uint64(6))
    lo64 = lo64 | (e0s[:, 0].astype(np.uint64) << np.uint64(7)) | (e1s[:, 0].astype(np.uint64) << np.uint64(14))
    lo64 = lo64 | (e0s[:, 1].astype(np.uint64) << np.uint64(21)) | (e1s[:, 1].astype(np.uint64) << np.uint64(28))
    lo64 = lo64 | (e0s[:, 2].astype(np.uint64) << np.uint64(35)) | (e1s[:, 2].astype(np.uint64) << np.uint64(42))
    lo64 = lo64 | (e0s[:, 3].astype(np.uint64) << np.uint64(49)) | (e1s[:, 3].astype(np.uint64) << np.uint64(56))
    lo64 = lo64 | (p0s.astype(np.uint64) << np.uint64(63))
    hi64 = p1s.astype(np.uint64) | (idx[:, 0].astype(np.uint64) << np.uint64(1))
    for k in range(1, 16):
        hi64 = hi64 | (idx[:, k].astype(np.uint64) << np.uint64(4 * k))
    out = np.empty((len(blk), 2), dtype="<u8"); out[:, 0] = lo64; out[:, 1] = hi64
    return out.tobytes()


def encode_bc(image: np.ndarray, fmt: str) -> bytes:
    """One mip level. BC1/BC3 through Pillow's DDS writer (RGB / RGBA uint8, H x W x C); BC5 from an (H, W, 2) uint8."""

    from PIL import Image

    a = np.asarray(image)
    h, w = a.shape[:2]
    ph, pw = max(4, ((h + 3) // 4) * 4), max(4, ((w + 3) // 4) * 4)
    if (ph, pw) != (h, w):                                     # small mips: pad to whole blocks by edge repeat
        a = np.pad(a, [(0, ph - h), (0, pw - w)] + [(0, 0)] * (a.ndim - 2), mode="edge")
    bw, bh = max(1, (w + 3) // 4), max(1, (h + 3) // 4)
    if fmt == "BC5":
        return _interleave_bc5(_bc4_blocks(a[..., 0]), _bc4_blocks(a[..., 1]))
    if fmt == "BC7":
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("BC7 needs an RGB(A) image")
        rgba = a if a.shape[2] == 4 else np.concatenate([a[..., :3], np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
        return _bc7_mode6_blocks(rgba)
    mode = "RGBA" if fmt == "BC3" else "RGB"
    im = Image.fromarray(a[..., :4] if mode == "RGBA" else a[..., :3], mode) if a.ndim == 3 else Image.fromarray(a).convert(mode)
    stream = io.BytesIO(); im.save(stream, format="DDS", pixel_format="DXT5" if fmt == "BC3" else "DXT1")
    data = stream.getvalue()
    need = bw * bh * (16 if fmt == "BC3" else 8)
    body = data[128:128 + need]
    if len(body) != need:
        raise ValueError("DDS body size mismatch")
    return body


def _interleave_bc5(x: bytes, y: bytes) -> bytes:
    a = np.frombuffer(x, np.uint8).reshape(-1, 8); b = np.frombuffer(y, np.uint8).reshape(-1, 8)
    return np.concatenate([a, b], 1).tobytes()


def _mip_chain(image: np.ndarray, srgb: bool, kind: str) -> list[np.ndarray]:
    """Box-filtered chain to 1 x 1: sRGB colour in linear light, normals renormalised, others plain."""

    levels = [np.asarray(image)]
    cur = np.asarray(image).astype(np.float64)
    if srgb:
        cur = (cur / 255.0) ** 2.2
    while cur.shape[0] > 1 or cur.shape[1] > 1:
        h, w = cur.shape[:2]
        nh, nw = max(1, h // 2), max(1, w // 2)
        fy, fx = (2 if h > 1 else 1), (2 if w > 1 else 1)      # a side that reached 1 stays 1
        c = cur[: nh * fy, : nw * fx]
        c = c.reshape(nh, fy, nw, fx, -1).mean((1, 3)) if c.ndim == 3 else c.reshape(nh, fy, nw, fx).mean((1, 3))
        cur = c
        if srgb:
            levels.append(np.clip(np.rint((np.clip(cur, 0, 1) ** (1 / 2.2)) * 255), 0, 255).astype(np.uint8))
        elif kind == "normal":
            v = cur[..., :2] / 127.5 - 1.0
            z = np.sqrt(np.clip(1 - (v ** 2).sum(-1, keepdims=True), 0, 1))
            n = np.concatenate([v, z], -1); n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
            levels.append(np.clip(np.rint((n[..., :2] + 1) * 127.5), 0, 255).astype(np.uint8))
        else:
            levels.append(np.clip(np.rint(cur), 0, 255).astype(np.uint8))
    return levels


def write_kwtex2(path: str | Path, image: np.ndarray, fmt: str, *, srgb: bool, kind: str = "color") -> dict[str, Any]:
    """Write a texture with its full mip chain. ``image`` is top-down uint8 (H, W, C): RGB(A) for BC1/BC3, XY for BC5."""

    a = np.asarray(image, np.uint8)
    h, w = a.shape[:2]
    if h & (h - 1) or w & (w - 1) or not (1 <= h <= 8192 and 1 <= w <= 8192):
        raise ValueError("KWTEX2 needs power-of-two sides up to 8192")
    code = TEX_FORMATS[fmt]
    if fmt == "BC5" and srgb:
        raise ValueError("BC5 is linear only")
    levels = _mip_chain(a[::-1], srgb, kind)                   # bottom-up rows once, before the chain
    records = []; blobs = []
    for lv in levels:
        blob = encode_bc(lv, fmt)
        expect = max(1, (lv.shape[1] + 3) // 4) * max(1, (lv.shape[0] + 3) // 4) * BLOCK_BYTES[code]
        if len(blob) != expect:
            raise ValueError("mip byte count mismatch")
        records.append((lv.shape[1], lv.shape[0], blob)); blobs.append(blob)
    table_off = HEADER_BYTES
    payload_off = _align16(table_off + 64 * len(records))
    off = payload_off; table = b""; body = bytearray()
    for (mw, mh, blob) in records:
        table += struct.pack("<IIQQQ", mw, mh, off, len(blob), 0) + hashlib.sha256(blob).digest()
        body += blob
        pad = _align16(off + len(blob)) - (off + len(blob))
        body += b"\0" * pad
        off += len(blob) + pad
    file_bytes = off
    payload = table + b"\0" * (payload_off - table_off - len(table)) + bytes(body)
    assert HEADER_BYTES + len(payload) == file_bytes
    head = struct.pack("<8sIIIIIIIIQQQ", KWTEX2_MAGIC, 1, HEADER_BYTES, code, 1 if srgb else 0, w, h, len(records), 0,
                       file_bytes, table_off, payload_off)
    head += hashlib.sha256(payload).digest(); head += b"\0" * (HEADER_BYTES - len(head))
    data = head + payload
    Path(path).write_bytes(data)
    return {"path": str(path), "bytes": file_bytes, "sha256": hashlib.sha256(data).hexdigest(), "format": fmt,
            "colour_space": "srgb" if srgb else "linear", "width": w, "height": h, "mips": len(records)}


def read_kwtex2_header(path: str | Path) -> dict[str, Any]:
    data = Path(path).read_bytes()
    magic, ver, hb, code, cs, w, h, mips, flags, fb, t_off, p_off = struct.unpack_from("<8sIIIIIIIIQQQ", data, 0)
    if magic != KWTEX2_MAGIC or ver != 1 or fb != len(data):
        raise ValueError("not a KWTEX2 v1 file")
    if hashlib.sha256(data[HEADER_BYTES:fb]).digest() != data[64:96]:
        raise ValueError("payload hash mismatch")
    recs = []
    for i in range(mips):
        mw, mh, off, cnt, _ = struct.unpack_from("<IIQQQ", data, t_off + 64 * i)
        dig = data[t_off + 64 * i + 32:t_off + 64 * i + 64]
        if hashlib.sha256(data[off:off + cnt]).digest() != dig:
            raise ValueError("mip hash mismatch at level %d" % i)
        recs.append({"width": mw, "height": mh, "offset": off, "bytes": cnt})
    return {"format": code, "colour_space": cs, "width": w, "height": h, "mips": recs}


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Validate the caps the spec names and write the manifest JSON."""

    m = dict(manifest)
    m.setdefault("magic", "KWORLD2"); m.setdefault("schema_version", 1)
    for key in ("build_id", "module_resref", "coordinates", "environment", "materials", "textures", "chunks"):
        if key not in m:
            raise ValueError("manifest missing " + key)
    if len(m["materials"]) > 4096 or len(m["textures"]) > 8192 or len(m["chunks"]) > 65536 or len(m.get("instances", [])) > 65536:
        raise ValueError("manifest over its caps")
    for c in m["chunks"]:
        p = Path(c["path"])
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("chunk paths must be relative, without traversal")
    text = json.dumps(m, indent=1)
    if len(text.encode("utf-8")) > (8 << 20):
        raise ValueError("manifest over 8 MiB")
    Path(path).write_text(text, encoding="utf-8")


__all__ = ("write_kwchunk2", "read_kwchunk2", "write_kwtex2", "read_kwtex2_header", "encode_bc", "write_manifest",
           "pack_rgb10a2", "unpack_rgb10a2", "CHUNK_FLAG_VERTEX_LIGHT", "TEX_FORMATS", "CHUNK_CAPS")
