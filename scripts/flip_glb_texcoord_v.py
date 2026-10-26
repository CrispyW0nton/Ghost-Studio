#!/usr/bin/env python3
"""Flip TEXCOORD_0 V values in a binary glTF for interchange diagnostics."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from pygltflib import GLTF2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gltf = GLTF2().load_binary(str(args.input))
    blob = bytearray(gltf.binary_blob() or b"")
    changed: set[int] = set()
    for mesh in gltf.meshes or []:
        for primitive in mesh.primitives or []:
            accessor_index = getattr(primitive.attributes, "TEXCOORD_0", None)
            if accessor_index is None or int(accessor_index) in changed:
                continue
            accessor = gltf.accessors[int(accessor_index)]
            view = gltf.bufferViews[int(accessor.bufferView)]
            if int(accessor.componentType) != 5126 or str(accessor.type) != "VEC2":
                raise ValueError(f"Unsupported TEXCOORD_0 accessor {accessor_index}")
            stride = int(view.byteStride or 8)
            start = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
            for row in range(int(accessor.count)):
                offset = start + row * stride
                u, v = struct.unpack_from("<ff", blob, offset)
                struct.pack_into("<ff", blob, offset, float(u), 1.0 - float(v))
            changed.add(int(accessor_index))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(args.output))
    print(f"Flipped {len(changed)} TEXCOORD_0 accessors: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
