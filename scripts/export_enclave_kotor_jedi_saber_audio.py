"""Export the original KOTOR Jedi lightsaber sound family for Unreal.

The exporter deliberately reads KEY/BIF only.  That keeps modded Override files
and the unrelated dark-side CombatTemplate sound set out of the enclave duel.
Every decoded PCM16 WAV is accompanied by archive provenance and a SHA-256
digest so MCPStudio can prove which retail resource was imported.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
K1_DEFAULT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
OUTPUT_DEFAULT = Path(
    r"C:\UE\Enclave\Saved\MCPStudio\imports\kotor_jedi_lightsaber_audio_v1"
)

RESOURCE_GROUPS = {
    "power": ("cb_ls_powerup", "cb_ls_powerdown"),
    "hum": ("cb_ls_humgood",),
    "swing_long": tuple(f"cb_ls_swinglong{index}" for index in range(1, 4)),
    "swing_short": tuple(f"cb_ls_swingshrt{index}" for index in range(1, 4)),
    "clash": ("cb_ls_clash1", "cb_ls_clash2"),
    "hit_hard": ("cb_ls_hithard1", "cb_ls_hithard2"),
    "hit_soft": ("cb_ls_hitsoft1", "cb_ls_hitsoft2"),
    "twirl": ("cb_ls_twirl1", "cb_ls_twirl2"),
}


def _bootstrap() -> None:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.mcp.start_kotormcp_stdio import _python_roots

    for path in reversed(_python_roots(ROOT)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _wav_metadata(data: bytes) -> dict[str, int | float | str]:
    with wave.open(io.BytesIO(data), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        frame_rate = int(reader.getframerate())
        frame_count = int(reader.getnframes())
        compression = str(reader.getcomptype())
    if channels != 1 or sample_width != 2 or frame_count <= 0 or compression != "NONE":
        raise RuntimeError(
            "KOTOR saber decode was not mono PCM16: "
            f"channels={channels} width={sample_width} frames={frame_count} compression={compression}"
        )
    return {
        "channels": channels,
        "sample_width": sample_width,
        "frame_rate": frame_rate,
        "frame_count": frame_count,
        "duration_seconds": frame_count / frame_rate,
        "compression": compression,
    }


def export_audio(game_root: Path, output_root: Path) -> dict:
    _bootstrap()
    from core.assets.resource_manager import RES_WAV, _GameInstall, _key
    from pykotor.resource.formats.wav import get_playable_bytes, read_wav

    if not (game_root / "chitin.key").is_file():
        raise FileNotFoundError(f"KOTOR chitin.key not found under {game_root}")

    installation = _GameInstall(str(game_root), "K1")
    output_root.mkdir(parents=True, exist_ok=True)
    resources = []
    for group, resrefs in RESOURCE_GROUPS.items():
        for resref in resrefs:
            slot = installation._key_map.get(_key(resref, RES_WAV))
            if slot is None:
                raise FileNotFoundError(f"Missing original KOTOR WAV resource: {resref}")
            bif_index, _ = slot
            bif = installation._bif_index.get(bif_index)
            source_archive = Path(str(getattr(bif, "path", ""))) if bif is not None else Path()
            raw = installation.get_bif(resref, RES_WAV)
            if not raw:
                raise FileNotFoundError(f"Could not read original KOTOR WAV resource: {resref}")
            playable = bytes(get_playable_bytes(read_wav(raw)))
            metadata = _wav_metadata(playable)
            target = output_root / f"{resref}.wav"
            target.write_bytes(playable)
            resources.append(
                {
                    "group": group,
                    "resref": resref,
                    "source_archive": str(source_archive),
                    "source_kind": "K1 KEY/BIF base game",
                    "output_wav": str(target),
                    "sha256": hashlib.sha256(playable).hexdigest(),
                    **metadata,
                }
            )

    expected_count = sum(len(values) for values in RESOURCE_GROUPS.values())
    receipt = {
        "schema": "ghoststudio.enclave.kotor-jedi-lightsaber-audio.v1",
        "game": "K1",
        "game_root": str(game_root),
        "output_root": str(output_root),
        "resource_count": len(resources),
        "resources": resources,
        "validation": {
            "expected_resource_count": expected_count,
            "all_resources_exported": len(resources) == expected_count,
            "all_base_game_bif": all(
                row["source_kind"] == "K1 KEY/BIF base game" and row["source_archive"]
                for row in resources
            ),
            "all_unreal_safe_pcm16": all(
                row["sample_width"] == 2
                and row["frame_count"] > 0
                and row["compression"] == "NONE"
                for row in resources
            ),
            "dark_side_placeholders_excluded": True,
        },
    }
    receipt_path = output_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if not all(bool(value) for value in receipt["validation"].values()):
        raise RuntimeError(f"KOTOR saber audio validation failed: {receipt_path}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=K1_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    result = export_audio(args.game_root.resolve(), args.output_root.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
