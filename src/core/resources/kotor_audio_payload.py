"""Headless Ghost Studio audio payload adapter using its PyKotor dependency."""
from pykotor.resource.formats.wav.wav_obfuscation import (
    DeobfuscationResult, detect_audio_format,
)


def _wave(data):
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _mp3(data):
    return data.startswith(b"ID3") or (
        len(data) >= 4 and data[0] == 255 and data[1] & 0xE0 == 0xE0
        and data[1] & 0x18 != 0x08 and data[1] & 6 == 2
        and data[2] >> 4 not in (0, 15) and data[2] & 12 != 12
    )


def playable_payload(data: bytes) -> tuple[bytes, str]:
    """Keep RIFF codec chunks intact for a downstream PCM/ADPCM decoder."""
    kind, offset = detect_audio_format(data)
    payload = data[offset:]
    if offset and _wave(payload):
        return payload, "wav"
    if offset and _mp3(payload) and kind == DeobfuscationResult.MP3_IN_WAV:
        return payload, "mp3"
    if _wave(data):
        return data, "wav"
    if _mp3(data):
        return data, "mp3"
    raise ValueError("Retail sound is neither supported RIFF/WAVE nor MPEG Layer III")
