"""Run a strict, sampled skin-deformation gate on one Aurora animation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.characters.animation_deformation_validator import (  # noqa: E402
    validate_animation_deformation,
)
from src.core.validation.viewport_validator import ViewportValidator  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mdl", required=True, type=Path)
    parser.add_argument("--mdx", type=Path, default=None)
    parser.add_argument("--game", choices=("K1", "K2", "k1", "k2"), default="K2")
    parser.add_argument("--animation", required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument(
        "--max-edge-stretch",
        type=float,
        default=3.5,
        help="Fail if any sampled triangle edge exceeds this bind-length ratio.",
    )
    parser.add_argument(
        "--min-edge-ratio",
        type=float,
        default=0.08,
        help="Fail if any sampled triangle edge falls below this bind-length ratio.",
    )
    parser.add_argument(
        "--max-floor-penetration",
        type=float,
        default=0.015,
        help="Fail if any sampled skinned vertex moves this far below the bind floor.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    if args.max_edge_stretch <= 1.0:
        raise SystemExit("--max-edge-stretch must be greater than 1")
    if not 0.0 < args.min_edge_ratio < 1.0:
        raise SystemExit("--min-edge-ratio must be between 0 and 1")
    if args.max_floor_penetration < 0.0:
        raise SystemExit("--max-floor-penetration cannot be negative")

    loader = ViewportValidator(output_dir=args.output.parent / ".deformation_loader")
    model = loader._load_mdl(
        args.mdl,
        mdx_path=args.mdx,
        game=args.game.upper(),
    )
    report = validate_animation_deformation(
        model,
        animations=[args.animation],
        samples_per_animation=args.samples,
    )
    if not report.samples:
        raise RuntimeError(
            f"Animation {args.animation!r} produced no deformation samples."
        )

    worst_stretch = max(report.samples, key=lambda sample: sample.max_edge_stretch)
    worst_collapse = min(report.samples, key=lambda sample: sample.min_edge_ratio)
    worst_floor = max(report.samples, key=lambda sample: sample.floor_penetration)
    hard_failures: list[str] = []
    if worst_stretch.max_edge_stretch > args.max_edge_stretch:
        hard_failures.append(
            "max_edge_stretch "
            f"{worst_stretch.max_edge_stretch:.6f} > {args.max_edge_stretch:.6f} "
            f"at {worst_stretch.time:.6f}s"
        )
    if worst_collapse.min_edge_ratio < args.min_edge_ratio:
        hard_failures.append(
            "min_edge_ratio "
            f"{worst_collapse.min_edge_ratio:.6f} < {args.min_edge_ratio:.6f} "
            f"at {worst_collapse.time:.6f}s"
        )
    if worst_floor.floor_penetration > args.max_floor_penetration:
        hard_failures.append(
            "max_floor_penetration "
            f"{worst_floor.floor_penetration:.6f} > "
            f"{args.max_floor_penetration:.6f} at {worst_floor.time:.6f}s"
        )

    accepted = report.ok and not hard_failures
    payload = {
        "accepted": accepted,
        "mdl": str(args.mdl.resolve()),
        "animation": args.animation,
        "sample_count": len(report.samples),
        "thresholds": {
            "max_edge_stretch": args.max_edge_stretch,
            "min_edge_ratio": args.min_edge_ratio,
            "max_floor_penetration": args.max_floor_penetration,
        },
        "extrema": {
            "max_edge_stretch": worst_stretch.max_edge_stretch,
            "max_edge_stretch_time": worst_stretch.time,
            "min_edge_ratio": worst_collapse.min_edge_ratio,
            "min_edge_ratio_time": worst_collapse.time,
            "max_floor_penetration": worst_floor.floor_penetration,
            "max_floor_penetration_time": worst_floor.time,
        },
        "hard_failures": hard_failures,
        "validator": report.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "accepted",
        "animation",
        "sample_count",
        "thresholds",
        "extrema",
        "hard_failures",
    )}, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
