from __future__ import annotations


def _dialogue():
    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import ResRef
    from pykotor.resource.generics.dlg import DLG, DLGAnimation, DLGEntry, DLGLink, DLGReply

    dialogue = DLG()
    dialogue.skippable = True
    dialogue.on_end = ResRef("dlg_finished")
    dialogue.on_abort = ResRef("dlg_aborted")

    greeting = DLGEntry()
    greeting.list_index = 0
    greeting.speaker = "OWNER"
    greeting.listener = "PLAYER"
    greeting.text = LocalizedString(1200)
    greeting.vo_resref = ResRef("owner_greeting")
    greeting.sound = ResRef("owner_greeting")
    greeting.script1 = ResRef("grant_intro")
    greeting.quest = "dan_training"
    greeting.quest_entry = 10
    greeting.camera_angle = 6
    greeting.camera_id = 3
    greeting.delay = 2100
    greeting.wait_flags = 3
    animation = DLGAnimation()
    animation.participant = "OWNER"
    animation.animation_id = 10040
    greeting.animations.append(animation)

    ask = DLGReply()
    ask.list_index = 0
    ask.text = LocalizedString.from_english("What happened here?")
    ask.script1 = ResRef("record_question")

    leave = DLGReply()
    leave.list_index = 1
    leave.text = LocalizedString.from_english("I should go.")

    answer = DLGEntry()
    answer.list_index = 1
    answer.speaker = "OWNER"
    answer.text = LocalizedString.from_english("The enclave is safe for now.")
    answer.vo_resref = ResRef("owner_answer")

    ask_link = DLGLink(ask)
    ask_link.active1 = ResRef("has_spoken")
    ask_link.active1_not = True
    ask_link.active2 = ResRef("is_padawan")
    ask_link.logic = False
    greeting.links.extend((ask_link, DLGLink(leave)))
    ask.links.append(DLGLink(answer))
    dialogue.starters.append(DLGLink(greeting))
    return dialogue


def test_manifest_preserves_kotor_graph_semantics_and_voice_provenance() -> None:
    from src.core.modules.kotor_dialogue_manifest import build_kotor_dialogue_manifest

    manifest = build_kotor_dialogue_manifest(
        _dialogue(),
        game="K1",
        conversation_resref="dan13_owner",
        tlk_lookup=lambda stringref: {1200: "Welcome to the enclave."}.get(stringref, ""),
        voice_assets={
            "owner_greeting": "/Game/Audio/KOTOR/dan13_owner/SW_owner_greeting",
            "owner_answer": "/Game/Audio/KOTOR/dan13_owner/SW_owner_answer",
        },
    )

    assert manifest["schema"] == "ghoststudio.kotor-dialogue-manifest.v2"
    assert manifest["source"]["conversation_resref"] == "dan13_owner"
    assert manifest["conversation"]["on_end"] == "dlg_finished"
    assert manifest["conversation"]["on_abort"] == "dlg_aborted"
    assert manifest["starters"] == ["starter:link:0"]

    nodes = {row["id"]: row for row in manifest["nodes"]}
    greeting = nodes["entry:0"]
    assert greeting["kind"] == "entry"
    assert greeting["speaker"] == "OWNER"
    assert greeting["listener"] == "PLAYER"
    assert greeting["text"] == "Welcome to the enclave."
    assert greeting["stringref"] == 1200
    assert greeting["voice_resref"] == "owner_greeting"
    assert greeting["voice_asset"] == "/Game/Audio/KOTOR/dan13_owner/SW_owner_greeting"
    assert greeting["camera"]["angle"] == 6
    assert greeting["camera"]["id"] == 3
    assert greeting["delay_ms"] == 2100
    assert greeting["wait_flags"] == 3
    assert greeting["animations"] == [{"participant": "OWNER", "animation_id": 10040}]
    assert greeting["actions"][0]["resref"] == "grant_intro"
    assert greeting["quest"] == {"tag": "dan_training", "entry": 10}

    links = {row["id"]: row for row in manifest["links"]}
    conditioned = links["entry:0:link:0"]
    assert conditioned["source"] == "entry:0"
    assert conditioned["target"] == "reply:0"
    assert conditioned["condition_logic"] == "AND"
    assert conditioned["conditions"] == [
        {"slot": "active1", "resref": "has_spoken", "negated": True, "parameters": []},
        {"slot": "active2", "resref": "is_padawan", "negated": False, "parameters": []},
    ]
    assert nodes["reply:0"]["actions"][0]["resref"] == "record_question"
    assert links["reply:0:link:0"]["target"] == "entry:1"

    validation = manifest["validation"]
    assert validation["node_count"] == 4
    assert validation["entry_count"] == 2
    assert validation["reply_count"] == 2
    assert validation["voiced_entry_count"] == 2
    assert validation["missing_voice_asset_resrefs"] == []


def test_manifest_falls_back_to_retail_generic_sound_when_streamed_vo_is_absent() -> None:
    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import ResRef
    from pykotor.resource.generics.dlg import DLG, DLGEntry, DLGLink

    from src.core.modules.kotor_dialogue_manifest import build_kotor_dialogue_manifest

    dialogue = DLG()
    greeting = DLGEntry()
    greeting.list_index = 0
    greeting.text = LocalizedString.from_english("Alien greeting")
    greeting.vo_resref = ResRef("missing_streamed_line")
    greeting.sound = ResRef("n_gmtwilek_grts")
    dialogue.starters.append(DLGLink(greeting))

    manifest = build_kotor_dialogue_manifest(
        dialogue,
        game="K1",
        conversation_resref="alien_vendor",
        voice_assets={"n_gmtwilek_grts": "/Game/Audio/SW_n_gmtwilek_grts"},
    )

    node = manifest["nodes"][0]
    assert node["voice_resref"] == "missing_streamed"
    assert node["sound_resref"] == "n_gmtwilek_grts"
    assert node["voice_asset_resref"] == "n_gmtwilek_grts"
    assert node["voice_asset"] == "/Game/Audio/SW_n_gmtwilek_grts"


def test_silent_glue_entry_sound_effect_is_not_reported_as_missing_voice() -> None:
    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import ResRef
    from pykotor.resource.generics.dlg import DLG, DLGEntry, DLGLink

    from src.core.modules.kotor_dialogue_manifest import build_kotor_dialogue_manifest

    dialogue = DLG()
    glue = DLGEntry()
    glue.list_index = 0
    glue.text = LocalizedString.from_english("")
    glue.sound = ResRef("gui_upgrade")
    dialogue.starters.append(DLGLink(glue))

    manifest = build_kotor_dialogue_manifest(
        dialogue,
        game="K1",
        conversation_resref="silent_glue",
        voice_assets={},
    )

    assert manifest["validation"]["voiced_entry_count"] == 0
    assert manifest["validation"]["missing_voice_asset_resrefs"] == []
