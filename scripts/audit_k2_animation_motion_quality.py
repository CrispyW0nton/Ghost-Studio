"""Measure temporal anatomy hazards in one KOTOR animation.

This is a read-only companion to the viewport renderer.  It evaluates every
delivery frame, reports the exact body nodes responsible for discontinuities,
checks elbow/knee bend-plane continuity, and measures contact penetration
relative to the model's rest floor.  Unlike the legacy preview ``passed``
flag, the measurements below are intended to be used as hard build gates.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.animation.animation_engine import evaluate_aurora_animation_pose
from src.core.game.kotor_loader import load_model_from_file
from src.core.geometry.model_data import GameVersion
from src.core.retargeting.reference_pose import compute_target_rest_transforms


BODY_NODES = {
    "rootdummy", "pelvis_g", "torso_g", "torsoupr_g",
    "lcollar_g", "lbicep_g", "lforearm_g", "lhand_g",
    "rcollar_g", "rbicep_g", "rforearm_g", "rhand_g",
    "lthigh_g", "lshin_g", "lfoot_g", "lfoott_g",
    "rthigh_g", "rshin_g", "rfoot_g", "rfoott_g",
    "necklwr_g", "neck_g", "hturn_g", "head_g",
}

CONTACT_NODES = ("lfoot_g", "lfoott_g", "rfoot_g", "rfoott_g")
CHAINS = {
    "left_arm": ("lbicep_g", "lforearm_g", "lhand_g"),
    "right_arm": ("rbicep_g", "rforearm_g", "rhand_g"),
    "left_leg": ("lthigh_g", "lshin_g", "lfoot_g"),
    "right_leg": ("rthigh_g", "rshin_g", "rfoot_g"),
}


def _quat_angle_degrees(first, second) -> float:
    a = [float(value) for value in first]
    b = [float(value) for value in second]
    an = math.sqrt(sum(value * value for value in a)) or 1.0
    bn = math.sqrt(sum(value * value for value in b)) or 1.0
    dot = abs(sum((a[index] / an) * (b[index] / bn) for index in range(4)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def _sub(a, b):
    return tuple(float(a[index]) - float(b[index]) for index in range(3))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(value) -> float:
    return math.sqrt(sum(float(component) ** 2 for component in value))


def _unit(value):
    length = _norm(value)
    if length <= 1e-9:
        return None
    return tuple(float(component) / length for component in value)


def _vector_angle_degrees(first, second) -> float:
    a = _unit(first)
    b = _unit(second)
    if a is None or b is None:
        return 0.0
    dot = max(-1.0, min(1.0, sum(a[index] * b[index] for index in range(3))))
    return math.degrees(math.acos(dot))


def _chain_normal(world_positions, chain):
    parent, joint, child = chain
    if any(name not in world_positions for name in chain):
        return None
    incoming = _unit(_sub(world_positions[joint], world_positions[parent]))
    outgoing = _unit(_sub(world_positions[child], world_positions[joint]))
    if incoming is None or outgoing is None:
        return None
    normal = _cross(incoming, outgoing)
    if _norm(normal) < 0.03:
        return None
    return _unit(normal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdl", required=True, type=Path)
    parser.add_argument("--mdx", type=Path)
    parser.add_argument("--animation", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_model_from_file(
        str(args.mdl),
        str(args.mdx or args.mdl.with_suffix(".mdx")),
        GameVersion.K2,
    )
    if model is None:
        raise RuntimeError(f"Could not load {args.mdl}")
    animation = next(
        (item for item in model.animations if str(item.name).lower() == args.animation.lower()),
        None,
    )
    if animation is None:
        raise RuntimeError(f"Animation {args.animation!r} was not found")

    frame_count = max(2, int(math.ceil(float(animation.length) * args.fps)) + 1)
    times = [min(float(animation.length), index / args.fps) for index in range(frame_count)]
    previous_rotations = {}
    previous_planes = {}
    per_node_max = {}
    body_events = []
    plane_events = []
    _rest_local, rest_world = compute_target_rest_transforms(model)
    rest_floor = min(
        rest_world[name].position[2]
        for name in CONTACT_NODES
        if name in rest_world
    )
    minimum_contact_z = float("inf")

    for frame_index, time_seconds in enumerate(times):
        pose = evaluate_aurora_animation_pose(model, animation, time_seconds)
        world_positions = {
            name.lower(): transform.position
            for name, transform in pose.world_transforms_by_node.items()
        }
        for name, transform in pose.local_transforms_by_node.items():
            key = name.lower()
            previous = previous_rotations.get(key)
            if previous is not None:
                angle = _quat_angle_degrees(previous, transform.rotation)
                current = per_node_max.get(key)
                if current is None or angle > current["degrees"]:
                    per_node_max[key] = {
                        "degrees": angle,
                        "frame": frame_index,
                        "time_seconds": time_seconds,
                    }
                if key in BODY_NODES and angle > 55.0:
                    body_events.append({
                        "node": name,
                        "degrees": angle,
                        "frame": frame_index,
                        "time_seconds": time_seconds,
                    })
            previous_rotations[key] = transform.rotation

        for contact in CONTACT_NODES:
            if contact in world_positions:
                minimum_contact_z = min(minimum_contact_z, float(world_positions[contact][2]))

        for chain_name, chain in CHAINS.items():
            normal = _chain_normal(world_positions, chain)
            if normal is None:
                continue
            previous = previous_planes.get(chain_name)
            if previous is not None:
                angle = _vector_angle_degrees(previous, normal)
                if angle > 80.0:
                    plane_events.append({
                        "chain": chain_name,
                        "degrees": angle,
                        "frame": frame_index,
                        "time_seconds": time_seconds,
                    })
            previous_planes[chain_name] = normal

    sorted_nodes = sorted(
        ({"node": node, **values} for node, values in per_node_max.items()),
        key=lambda item: item["degrees"],
        reverse=True,
    )
    body_nodes = [item for item in sorted_nodes if item["node"] in BODY_NODES]
    report = {
        "schema": 1,
        "mdl": str(args.mdl.resolve()),
        "animation": animation.name,
        "duration_seconds": float(animation.length),
        "fps": float(args.fps),
        "sample_count": len(times),
        "rest_floor_z": rest_floor,
        "minimum_contact_z": minimum_contact_z,
        "max_floor_penetration_meters": max(0.0, rest_floor - minimum_contact_z),
        "max_body_frame_step_degrees": body_nodes[0]["degrees"] if body_nodes else 0.0,
        "body_frame_step_violation_count": len(body_events),
        "max_bend_plane_discontinuity_degrees": max(
            (item["degrees"] for item in plane_events), default=0.0
        ),
        "bend_plane_violation_count": len(plane_events),
        "top_body_steps": body_nodes[:20],
        "top_all_node_steps": sorted_nodes[:30],
        "body_step_violations": body_events[:100],
        "bend_plane_violations": plane_events[:100],
        "passed": not body_events
        and not plane_events
        and max(0.0, rest_floor - minimum_contact_z) <= 0.015,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
