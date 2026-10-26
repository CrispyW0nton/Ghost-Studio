"""Convert a KOTOR/Ghost Studio ASCII FBX to Autodesk binary FBX.

The direct Ghost Studio FBX writer intentionally emits the portable ASCII
flavour.  Blender's FBX importer rejects ASCII FBX, so animation packaging must
cross a format-only boundary before Blender sees it.  Maya's Autodesk FBX plug-
in performs that conversion without recreating any skeleton transforms.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import maya.standalone


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as cmds
        import maya.mel as mel

        if not cmds.pluginInfo("fbxmaya", query=True, loaded=True):
            cmds.loadPlugin("fbxmaya", quiet=True)

        cmds.file(new=True, force=True)
        mel.eval("FBXResetImport")
        cmds.file(
            str(args.input),
            i=True,
            type="FBX",
            ignoreVersion=True,
            mergeNamespacesOnClash=False,
            namespace=":",
            options="fbx",
            preserveReferences=True,
        )

        all_nodes = cmds.ls(long=True) or []
        type_counts = Counter(cmds.nodeType(node) for node in all_nodes)
        joints = cmds.ls(type="joint", long=True) or []
        meshes = cmds.ls(type="mesh", long=True) or []
        animation_curves = cmds.ls(type="animCurve", long=True) or []
        print(json.dumps({"maya_node_types": dict(sorted(type_counts.items()))}, indent=2))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        mel.eval("FBXResetExport")
        mel.eval("FBXExportInAscii -v false")
        mel.eval("FBXExportBakeComplexAnimation -v true")
        mel.eval("FBXExportConstraints -v false")
        mel.eval("FBXExportSkins -v true")
        escaped = str(args.output).replace("\\", "/").replace('"', '\\"')
        mel.eval(f'FBXExport -f "{escaped}"')

        signature = args.output.read_bytes()[:23]
        payload = {
            "schema": "ghostrigger.fbx.format-conversion.v1",
            "input": str(args.input),
            "output": str(args.output),
            "output_bytes": args.output.stat().st_size if args.output.is_file() else 0,
            "binary_signature": signature.hex(),
            "joint_count": len(joints),
            "mesh_shape_count": len(meshes),
            "animation_curve_count": len(animation_curves),
            "maya_node_types": dict(sorted(type_counts.items())),
            "transform_policy": "format_only_no_skeleton_reconstruction",
            "all_checks_passed": bool(
                args.output.is_file()
                and args.output.stat().st_size > 100_000
                and signature.startswith(b"Kaydara FBX Binary")
                and len(all_nodes) >= 20
            ),
        }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if payload["all_checks_passed"] else 2
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    raise SystemExit(main())
