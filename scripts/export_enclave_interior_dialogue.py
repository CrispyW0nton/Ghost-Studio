"""Extract canonical K1 conversation graphs and Unreal-safe original voice audio."""

from __future__ import annotations

import argparse
import array
import contextlib
import io
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K1_DEFAULT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\swkotor")
EXPORT_DEFAULT = Path(r"C:\UE\Enclave\Saved\MCPStudio\exports\enclave_characters_v1")
ROSTER_DEFAULT = EXPORT_DEFAULT / "enclave_interior_roster_v1.json"
UNREAL_DIALOGUE_ROOT = "/Game/MCPStudio/Dialogue/KOTOR"
UNREAL_PROJECT_ROOT = Path(r"C:\UE\Enclave")
UNREAL_RUNTIME_DIALOGUE_ROOT = UNREAL_PROJECT_ROOT / "Content" / "MCPStudio" / "Dialogue" / "KOTOR"

# These are verbatim K1 NPC entries selected for ambient use.  They avoid
# starting quests, spending credits, changing plot state, or implying that the
# player made a dialogue choice.  Entries remain attributable to their source
# DLG/StringRef/VO ResRef in the receipt.
PREFERRED_LINE_BY_CONVERSATION = {
    "dan13_ahlan": "These two families have been settled here for some years now...",
    "dan13_belaya": "Greetings, young Padawan. I trust your training goes well.",
    "dan13_crattis": "It's always good to see another ally in the fight against darkness and corruption.",
    "dan13_dorak": "May the Force be with you.",
    "dan13_jedi": "Dantooine is one of our last refuges. The Sith hunt us everywhere.",
    "dan14_jedi": "It is relaxing to come outside and enjoy the fresh air once in a while. It relieves stress.",
    "dan13_jdroid": "The Council has decreed you may come and go as you please.",
    "dan13_karal": "I greet you, human. How may this outpost of Aratech Droids serve your needs?",
    "ebo46_lurarka": "As you desire. I bid you good day, sentient.",
    "dan13_pdroid": "I am sorry, Padawan, I am unable to render assistance at this time.",
    "dan13_vandar": "May the Force be with you.",
    "dan13_vrook": "Keep in mind the danger of the dark side, lest you fall into its grasp.",
    "dan13_zhar": "There is no ignorance...",
    "dan13_solaa": "Care to have a hand of Pazaak with me?",
    "k_hdroid2_dialog": "I cannot render assistance, citizen. Move along.",
    "dan13_droid": "Beep beep beep-oop!",
    "dan13_deesra": "I only wanted to wish you well, and to warn you to be careful. There are many evils out there, many dangers. The Sith are not the only minions of the dark side.",
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


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _voice_index(game_root: Path) -> dict[str, Path]:
    return {
        path.stem.casefold(): path
        for path in (game_root / "streamwaves").rglob("*.wav")
    }


def _write_unreal_pcm16(source: Path, target: Path) -> None:
    """Transcode KOTOR's unsigned 8-bit PCM voices to Unreal-safe PCM16."""

    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        compression = reader.getcomptype()
        frames = reader.readframes(reader.getnframes())
    # KOTOR's streamed voices commonly use a RIFF/WAV compatibility wrapper
    # whose data size is zero while an MPEG audio stream begins immediately
    # after the empty data header.  Passing the whole wrapper to FFmpeg makes
    # it trust the PCM-u8 fmt chunk and decode MPEG bytes as static.  Detect
    # the embedded frame sync and feed only that payload to the MP3 demuxer.
    if not frames and source.stat().st_size > 44:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                f"KOTOR WAV has a zero-length RIFF data header and FFmpeg is unavailable: {source}"
            )
        raw = source.read_bytes()
        data_offset = raw.find(b"data")
        payload_start = data_offset + 8 if data_offset >= 0 else 0
        mpeg_start = next(
            (
                index
                for index in range(payload_start, max(payload_start, len(raw) - 1))
                if raw[index] == 0xFF and (raw[index + 1] & 0xE0) == 0xE0
            ),
            -1,
        )
        if mpeg_start < 0:
            raise RuntimeError(
                f"KOTOR WAV has no PCM frames and no embedded MPEG frame sync: {source}"
            )
        embedded_mpeg = raw[mpeg_start:]
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "mp3",
                "-i",
                "pipe:0",
                "-c:a",
                "pcm_s16le",
                str(target),
            ],
            check=True,
            input=embedded_mpeg,
        )
        with wave.open(str(target), "rb") as converted:
            if (
                converted.getsampwidth() != 2
                or converted.getnframes() <= 0
                or converted.getcomptype() != "NONE"
            ):
                raise RuntimeError(f"FFmpeg produced an invalid Unreal WAV: {target}")
        return
    if compression != "NONE":
        raise RuntimeError(f"Unsupported compressed KOTOR WAV: {source}")
    if sample_width == 1:
        samples = array.array("h", ((sample - 128) << 8 for sample in frames))
        if sys.byteorder != "little":
            samples.byteswap()
        pcm16 = samples.tobytes()
    elif sample_width == 2:
        pcm16 = frames
    else:
        raise RuntimeError(
            f"Unsupported KOTOR WAV sample width {sample_width * 8}-bit: {source}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(pcm16)


def _validate_unreal_pcm16(path: Path) -> dict:
    with wave.open(str(path), "rb") as reader:
        result = {
            "channels": reader.getnchannels(),
            "sample_width": reader.getsampwidth(),
            "frame_rate": reader.getframerate(),
            "frame_count": reader.getnframes(),
            "compression": reader.getcomptype(),
        }
    if (
        result["sample_width"] != 2
        or result["frame_count"] <= 0
        or result["compression"] != "NONE"
    ):
        raise RuntimeError(f"Voice conversion did not produce Unreal-safe PCM16: {path}")
    return result


def _canonical_audio_resrefs(dialogue) -> list[str]:
    values = []
    for node in dialogue.all_entries():
        for value in (
            str(getattr(node, "vo_resref", "") or "").strip(),
            str(getattr(node, "sound", "") or "").strip(),
        ):
            if value and value.casefold() not in {item.casefold() for item in values}:
                values.append(value)
    return values


def _load_dialogue_payload(installation, templates, conversation: str, resource_type, search_location):
    payload = templates.resource(conversation, resource_type)
    if payload:
        return bytes(payload)
    result = installation.resource(
        conversation,
        resource_type,
        order=[search_location.OVERRIDE, search_location.MODULES, search_location.CHITIN],
        module_root="danm13",
    )
    return bytes(result.data) if result and result.data else None


def _runtime_manifest_relative_path(conversation: str) -> str:
    return f"Content/MCPStudio/Dialogue/KOTOR/{conversation}/conversation.json"


def export_dialogue_v2(
    game_root: Path,
    roster_path: Path,
    output_root: Path,
    runtime_dialogue_root: Path = UNREAL_RUNTIME_DIALOGUE_ROOT,
) -> dict:
    """Export every reachable canonical DLG node/link plus each referenced voice."""

    _bootstrap()
    from pykotor.extract.capsule import LazyCapsule
    from pykotor.extract.installation import Installation, SearchLocation
    from pykotor.resource.formats.tlk import read_tlk
    from pykotor.resource.generics.dlg import read_dlg
    from pykotor.resource.type import ResourceType as RT

    from src.core.modules.kotor_dialogue_manifest import build_kotor_dialogue_manifest

    roster = json.loads(roster_path.read_text(encoding="utf-8"))["creatures"]
    templates = LazyCapsule(game_root / "modules" / "danm13_s.rim")
    talk_table = read_tlk(game_root / "dialog.tlk")
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        installation = Installation(game_root)
    voices = _voice_index(game_root)
    dialogue_root = output_root / "dialogue_v2"
    dialogue_root.mkdir(parents=True, exist_ok=True)

    conversations: dict[str, dict] = {}
    for conversation in dict.fromkeys(
        str(creature["conversation_resref"]).casefold() for creature in roster
    ):
        payload = _load_dialogue_payload(
            installation,
            templates,
            conversation,
            RT.DLG,
            SearchLocation,
        )
        if not payload:
            raise FileNotFoundError(f"Missing canonical conversation {conversation}.dlg")
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            dialogue = read_dlg(payload)

        conversation_root = dialogue_root / conversation
        audio_root = conversation_root / "audio"
        conversation_root.mkdir(parents=True, exist_ok=True)
        voice_assets: dict[str, str] = {}
        audio_receipts = []
        for audio_resref in _canonical_audio_resrefs(dialogue):
            source = voices.get(audio_resref.casefold())
            if source is None:
                continue
            target = audio_root / f"{audio_resref}.wav"
            _write_unreal_pcm16(source, target)
            wav = _validate_unreal_pcm16(target)
            asset_path = f"{UNREAL_DIALOGUE_ROOT}/{conversation}/Audio/SW_{audio_resref}"
            voice_assets[audio_resref.casefold()] = asset_path
            audio_receipts.append(
                {
                    "resref": audio_resref,
                    "source": str(source),
                    "wav": str(target),
                    "unreal_asset": asset_path,
                    **wav,
                }
            )

        def tlk_lookup(stringref: int) -> str:
            return (
                _compact_text(talk_table.entries[stringref].text)
                if 0 <= stringref < len(talk_table.entries)
                else ""
            )

        manifest = build_kotor_dialogue_manifest(
            dialogue,
            game="K1",
            conversation_resref=conversation,
            tlk_lookup=tlk_lookup,
            voice_assets=voice_assets,
        )
        manifest["audio"] = audio_receipts
        manifest_path = conversation_root / "conversation.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        runtime_manifest_path = runtime_dialogue_root / conversation / "conversation.json"
        runtime_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        conversations[conversation] = {
            "manifest": str(manifest_path),
            "runtime_manifest": _runtime_manifest_relative_path(conversation),
            "runtime_manifest_source": str(runtime_manifest_path),
            "unreal_manifest_asset": f"{UNREAL_DIALOGUE_ROOT}/{conversation}/DA_{conversation}",
            "node_count": manifest["validation"]["node_count"],
            "link_count": manifest["validation"]["link_count"],
            "voiced_entry_count": manifest["validation"]["voiced_entry_count"],
            "exported_voice_count": len(audio_receipts),
            "missing_voice_asset_resrefs": manifest["validation"]["missing_voice_asset_resrefs"],
        }

    characters = [
        {
            "placement_index": int(creature["placement_index"]),
            "utc_resref": str(creature["utc_resref"]),
            "display_name": str(creature["display_name"]),
            "conversation_resref": str(creature["conversation_resref"]).casefold(),
            **conversations[str(creature["conversation_resref"]).casefold()],
        }
        for creature in roster
    ]
    result = {
        "schema": "ghoststudio.enclave-interior-dialogue.v2",
        "module": "m13aa",
        "character_count": len(characters),
        "conversation_count": len(conversations),
        "characters": characters,
        "conversations": conversations,
        "validation": {
            "all_conversations_exported": len(conversations) == len(
                {str(creature["conversation_resref"]).casefold() for creature in roster}
            ),
            "total_nodes": sum(row["node_count"] for row in conversations.values()),
            "total_links": sum(row["link_count"] for row in conversations.values()),
            "total_exported_voices": sum(row["exported_voice_count"] for row in conversations.values()),
            "conversations_with_missing_voice_assets": {
                key: row["missing_voice_asset_resrefs"]
                for key, row in conversations.items()
                if row["missing_voice_asset_resrefs"]
            },
        },
    }
    receipt = output_root / "enclave_interior_dialogue_v2.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def export_dialogue(game_root: Path, roster_path: Path, output_root: Path) -> dict:
    _bootstrap()
    from pykotor.extract.capsule import LazyCapsule
    from pykotor.extract.installation import Installation, SearchLocation
    from pykotor.resource.formats.tlk import read_tlk
    from pykotor.resource.generics.dlg import read_dlg
    from pykotor.resource.type import ResourceType as RT

    roster = json.loads(roster_path.read_text(encoding="utf-8"))["creatures"]
    templates = LazyCapsule(game_root / "modules" / "danm13_s.rim")
    talk_table = read_tlk(game_root / "dialog.tlk")
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
        installation = Installation(game_root)
    voices = _voice_index(game_root)

    rows = []
    for creature in roster:
        conversation = creature["conversation_resref"].casefold()
        desired = PREFERRED_LINE_BY_CONVERSATION.get(conversation, "")
        payload = templates.resource(conversation, RT.DLG)
        if not payload:
            result = installation.resource(
                conversation,
                RT.DLG,
                order=[SearchLocation.OVERRIDE, SearchLocation.MODULES, SearchLocation.CHITIN],
                module_root="danm13",
            )
            payload = bytes(result.data) if result and result.data else None
        if not payload:
            raise FileNotFoundError(f"Missing canonical conversation {conversation}.dlg")

        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            dialogue = read_dlg(payload)
        candidates = []
        for node in dialogue.all_entries():
            stringref = int(node.text.stringref)
            text = (
                _compact_text(talk_table.entries[stringref].text)
                if 0 <= stringref < len(talk_table.entries)
                else ""
            )
            if not text:
                continue
            candidates.append(
                {
                    "text": text,
                    "stringref": stringref,
                    "vo_resref": str(node.vo_resref),
                }
            )
        selected = next((row for row in candidates if row["text"] == desired), None)
        if selected is None:
            raise RuntimeError(
                f"The reviewed ambient line drifted or is missing from {conversation}.dlg: {desired!r}"
            )

        audio_output = ""
        source = voices.get(selected["vo_resref"].casefold())
        if source:
            character_audio = output_root / creature["utc_resref"] / "audio"
            character_audio.mkdir(parents=True, exist_ok=True)
            target = character_audio / "passing_dialogue.wav"
            _write_unreal_pcm16(source, target)
            audio_output = str(target)

        rows.append(
            {
                "placement_index": creature["placement_index"],
                "utc_resref": creature["utc_resref"],
                "display_name": creature["display_name"],
                "conversation_resref": conversation,
                "text": selected["text"],
                "stringref": selected["stringref"],
                "vo_resref": selected["vo_resref"],
                "voice_source": str(source) if source else "",
                "voice_wav": audio_output,
                "has_canonical_voice": bool(audio_output),
            }
        )

    payload = {
        "schema": "ghoststudio.enclave-interior-dialogue.v1",
        "module": "m13aa",
        "character_count": len(rows),
        "voiced_character_count": sum(1 for row in rows if row["has_canonical_voice"]),
        "characters": rows,
    }
    receipt = output_root / "enclave_interior_dialogue_v1.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-root", type=Path, default=K1_DEFAULT)
    parser.add_argument("--roster", type=Path, default=ROSTER_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=EXPORT_DEFAULT)
    args = parser.parse_args()
    payload = export_dialogue_v2(args.game_root, args.roster, args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
