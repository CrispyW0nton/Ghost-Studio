"""Maya-side binary normalization of one Ghost Studio FBX for Unreal Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import maya.cmds as cmds
import maya.mel as mel
import maya.standalone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def mel_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    maya.standalone.initialize(name="python")
    try:
        cmds.file(new=True, force=True)
        if not cmds.pluginInfo("fbxmaya", query=True, loaded=True):
            cmds.loadPlugin("fbxmaya", quiet=True)
        mel.eval("FBXResetImport")
        mel.eval("FBXImportFillTimeline -v true")
        mel.eval("FBXImportGenerateLog -v false")
        mel.eval(f'FBXImport -f "{mel_path(args.input)}"')

        joints = cmds.ls(type="joint", long=True) or []
        meshes = cmds.ls(type="mesh", long=True) or []
        skins = cmds.ls(type="skinCluster", long=True) or []
        curves = cmds.ls(type="animCurve", long=True) or []
        if not joints or not meshes or not skins:
            raise RuntimeError(
                "Maya FBX normalization requires joints, meshes, and skin clusters; "
                f"joints={len(joints)} meshes={len(meshes)} skins={len(skins)}"
            )
        if not curves:
            raise RuntimeError("Maya FBX normalization found no animation curves")

        start = float(cmds.playbackOptions(query=True, minTime=True))
        end = float(cmds.playbackOptions(query=True, maxTime=True))
        mel.eval("FBXResetExport")
        mel.eval('FBXExportFileVersion -v "FBX202000"')
        mel.eval("FBXExportInAscii -v false")
        mel.eval("FBXExportSkins -v true")
        mel.eval("FBXExportShapes -v true")
        mel.eval("FBXExportInputConnections -v true")
        mel.eval("FBXExportConstraints -v false")
        mel.eval("FBXExportCameras -v false")
        mel.eval("FBXExportLights -v false")
        mel.eval("FBXExportBakeComplexAnimation -v true")
        mel.eval(f"FBXExportBakeComplexStart -v {start}")
        mel.eval(f"FBXExportBakeComplexEnd -v {end}")
        mel.eval("FBXExportBakeComplexStep -v 1")
        mel.eval("FBXExportQuaternion -v resample")
        mel.eval("FBXExportAxisConversionMethod convertAnimation")
        mel.eval(f'FBXExport -f "{mel_path(args.output)}"')
        if not args.output.is_file() or args.output.stat().st_size < 1024:
            raise RuntimeError(f"Maya did not produce a valid FBX: {args.output}")

        report = {
            "schema": "ghoststudio.maya-unreal-fbx-normalization.v1",
            "input": str(args.input),
            "output": str(args.output),
            "output_bytes": args.output.stat().st_size,
            "joint_count": len(joints),
            "mesh_count": len(meshes),
            "skin_cluster_count": len(skins),
            "animation_curve_count": len(curves),
            "frame_range": [start, end],
        }
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("GHOSTSTUDIO_UNREAL_FBX_NORMALIZATION=" + json.dumps(report, sort_keys=True))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    raise SystemExit(main())
