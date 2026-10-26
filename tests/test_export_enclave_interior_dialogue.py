from __future__ import annotations

import struct
import wave
from pathlib import Path

from scripts import export_enclave_interior_dialogue as exporter


def test_canonical_audio_resrefs_include_streamed_voice_and_generic_sound() -> None:
    class Node:
        vo_resref = "nm13aacrat07031_"
        sound = "n_gmtwilek_grts"

    class Dialogue:
        def all_entries(self):
            return [Node()]

    assert exporter._canonical_audio_resrefs(Dialogue()) == [
        "nm13aacrat07031_",
        "n_gmtwilek_grts",
    ]


def test_runtime_manifest_path_is_project_relative_and_packageable() -> None:
    assert exporter._runtime_manifest_relative_path("dan13_belaya") == (
        "Content/MCPStudio/Dialogue/KOTOR/dan13_belaya/conversation.json"
    )


def test_zero_length_kotor_riff_decodes_embedded_mpeg_payload(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "kotor_voice.wav"
    target = tmp_path / "unreal_voice.wav"
    source.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 22050, 22050, 1, 8)
        + b"data"
        + struct.pack("<I", 0)
        + b"\xff\xfb\x38\xc4embedded-mp3"
    )
    calls = []

    def fake_run(command, check, input):
        calls.append((command, check, input))
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(22050)
            writer.writeframes(struct.pack("<hhhh", -8192, 0, 8192, 0))

    monkeypatch.setattr(exporter.shutil, "which", lambda name: "ffmpeg.exe")
    monkeypatch.setattr(exporter.subprocess, "run", fake_run)

    exporter._write_unreal_pcm16(source, target)

    assert calls and calls[0][1] is True
    assert calls[0][0][0] == "ffmpeg.exe"
    assert calls[0][0][calls[0][0].index("-f") + 1] == "mp3"
    assert calls[0][0][calls[0][0].index("-i") + 1] == "pipe:0"
    assert calls[0][2] == b"\xff\xfb\x38\xc4embedded-mp3"
    with wave.open(str(target), "rb") as converted:
        assert converted.getsampwidth() == 2
        assert converted.getnframes() == 4
