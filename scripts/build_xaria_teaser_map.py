"""Build the KOTOR II Xaria teaser encounter as a deterministic Map Studio candidate.

The module intentionally reuses only assets from the user's installed KOTOR II
library.  It does not stage or install anything.  The emitted module carries
the byte-verified production ``xar_*`` encounter spine accepted in PLCaa and
hands its encounter actor to the production ``xaria.dlg`` recruitment flow.
Legacy private ``xt_*`` authoring helpers remain available for historical test
coverage but are excluded from the packed module.  Xaria's custom models,
production dialogue/scripts, global 2DA rows, and runtime patches remain
explicit external dependencies.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mcp.start_kotormcp_stdio import _python_roots  # noqa: E402

for item in reversed(list(_python_roots(ROOT))):
    if item.exists() and str(item) not in sys.path:
        sys.path.insert(0, str(item))


MODULE_ROOT = "xartease"
VOICE_STREAM_ID = "997"
VOICE_DIALOGUE_RESREF = "xaria"
SOURCE_GAME = "K2"
SOURCE_MODULE = "402dxn"
SOURCE_ROOM = "402dxna"
ROOM_PIECE_ID = "k2_402dxn_402dxna"
AUTHOR = "LordVaderCW"
ARTIFACT_DIR = ROOT / "artifacts" / "xaria_teaser"
KMAP_PATH = ARTIFACT_DIR / f"{MODULE_ROOT}.kmap"
MOD_PATH = ARTIFACT_DIR / f"{MODULE_ROOT}.mod"
PROOF_PATH = ARTIFACT_DIR / "structural_proof.json"
MANIFEST_PATH = ARTIFACT_DIR / "xaria_teaser_manifest.json"
BLUEPRINT_PATH = ARTIFACT_DIR / "xaria_teaser_spatial_blueprint.png"
STORYBOARD_PATH = ARTIFACT_DIR / "xaria_teaser_showcase_storyboard.png"
UI_WORKFLOW_PATH = ARTIFACT_DIR / "GHOST_STUDIO_MANUAL_WORKFLOW.md"
XARIA_CANDIDATE_ROOT_ENV = "XARIA_CANDIDATE_ROOT"

ENTRY_POINT = (-7.0, -58.0, 2.651398)
HERO_CLEARING_CENTER = (-5.0, -18.0, 9.808109)
EXIT_POINT = (1.5, -1.2, 9.648286)
ROUTE_POINTS = (
    (-7.0, -58.0),
    (-13.25, -52.0),
    (-21.25, -45.5),
    (-25.25, -36.5),
    (-22.0, -31.5),
    (-17.5, -26.5),
    (-11.0, -24.5),
    (-5.0, -18.0),
)
EXIT_ROUTE_POINTS = (
    (-5.0, -18.0),
    (-2.0, -10.0),
    (0.0, -7.0),
    (1.5, -1.2),
)

WORLD_LIGHTING = {
    "profile": "custom",
    "sun_ambient": (43, 57, 45),
    "sun_diffuse": (78, 93, 70),
    "dynamic_ambient": (31, 46, 37),
    "shadow_opacity": 196,
    "sun_shadows": True,
    "fog_enabled": True,
    "fog_color": (35, 54, 41),
    "fog_near": 7.0,
    "fog_far": 52.0,
}

PRIVATE_XARIA_TEMPLATE = "xt_xaria"
PRIVATE_XARIA_TAG = "XT_Xaria"
PRIVATE_XARIA_FACTION_ID = 2
PRIVATE_WRAID_TEMPLATES = ("xt_wr1", "xt_wr2", "xt_wr3")
PRIVATE_WRAID_TAGS = ("XT_Wraid_1", "XT_Wraid_2", "XT_Wraid_3")
# Keep the showcase targets neutral until the cinematic conversation has
# actually started.  Packing them as Hostile_1 lets ordinary perception/combat
# steal control from the director before the first silent combat node runs.
PRIVATE_WRAID_FACTION_ID = 5
PRIVATE_TRIGGER_TEMPLATE = "xt_intro"
PRIVATE_TRIGGER_TAG = "XT_Intro_Trigger"
PRIVATE_DIRECTOR_TEMPLATE = "xt_director"
PRIVATE_DIRECTOR_TAG = "XT_Director"
PRIVATE_DIALOGUE = "xt_dlg"
PRIVATE_BLADE = "xt_blade"
PRODUCTION_DIALOGUE = "xaria"
PRODUCTION_GLOBAL_STATE = "KPM_XARIA_STATE"
PRODUCTION_GLOBAL_SLOT = "KPM_XARIA_SLOT"
PRIVATE_SCHEMA = 14
PRIVATE_POWER_ROWS = (287, 290, 291)
PRIVATE_POWER_TOKENS = (1, 2, 3)
PRIVATE_CAMERA_POLL_SECONDS = 1
PRIVATE_BEAT_FINISH_SECONDS = (2.60, 2.80, 3.10)
PRIVATE_DEATH_SETTLE_SECONDS = 0.20
PRIVATE_DEATH_RETRY_SECONDS = 0.25
PRIVATE_CAMERA_PREROLL_SECONDS = 1.25
PRIVATE_MAX_COMBAT_TIMELINE_SECONDS = (
    sum(PRIVATE_BEAT_FINISH_SECONDS)
    + 3 * (PRIVATE_DEATH_SETTLE_SECONDS + PRIVATE_DEATH_RETRY_SECONDS)
    + 3 * PRIVATE_CAMERA_PREROLL_SECONDS
)
# Retail K2 stores creature LocalNumbers in byte-sized slots 12..28. Keep the
# private encounter inside that range and store compact proof tokens instead
# of spells.2da rows (which are currently 287+ and cannot fit in a byte).
PRIVATE_ACTIVE_BEAT_LOCAL = 21
PRIVATE_DIRECTOR_TARGET_LOCAL = 22
PRIVATE_ENTRY_PATH_LOCAL = 23
PRIVATE_SCHEMA_LOCAL = 24
PRIVATE_SEQUENCE_PROOF_LOCAL = 25
PRIVATE_POWER_PROOF_LOCAL = 26
PRIVATE_WATCHDOG_LOCAL = 27
PRIVATE_STATE_LOCAL = 28
# Retail's documented creature LocalBoolean range is 20..63.
PRIVATE_CAMERA_HANDOFF_LOCAL = 60
PRIVATE_RECRUIT_ORIGIN_LOCAL = 62
PRIVATE_DIALOGUE_STARTED_LOCAL = 61
PRIVATE_DIALOGUE_PENDING_LOCAL = 63
DIRECTOR_POWER_SCRIPT_RESREFS = (
    "kxar_d_mamb",
    "kxar_d_ilight",
    "kxar_d_idrain",
)
PRODUCTION_XARIA_TEMPLATE = "xar_enc"
PRODUCTION_XARIA_TAG = "Xaria_Encounter"
PRODUCTION_WRAID_TEMPLATES = ("xar_nydak1", "xar_nydak2", "xar_nydak3")
PRODUCTION_WRAID_TAGS = ("Xaria_Wraid_1", "Xaria_Wraid_2", "Xaria_Wraid_3")
PRODUCTION_TRIGGER_TEMPLATE = "xar_introtrig"
PRODUCTION_TRIGGER_TAG = "Xaria_Intro_Trigger"
PRODUCTION_DIRECTOR_TEMPLATE = "xar_director"
PRODUCTION_DIRECTOR_TAG = "Xaria_Director"
PRODUCTION_TIMELINE_TEMPLATE = "xar_clock"
PRODUCTION_TIMELINE_TAG = "Xaria_Timeline"
PRODUCTION_CAMERA_DIALOGUE = "xar_cam"
PRODUCTION_BLADE = "xaria_blade"
PRODUCTION_SOUND_OBJECTS = (
    ("kxar_mistsnd", "kxar_mistsnd"),
    ("kxar_lightsnd", "kxar_lightsnd"),
    ("kxar_drainsnd", "kxar_drainsnd"),
)
PRODUCTION_ENCOUNTER_NCS = (
    "kxar_start",
    "kxar_cam1",
    "kxar_show",
    "kxar_showfail",
    "kxar_poll",
    "kxar_finish",
    "kxar_camend",
    "kxar_heart",
    "kxar_click",
    "kxar_enddlg",
    "kxar_wheart",
    "kxar_wspell",
    "kxar_wdeath",
    "kxar_ai_atk",
    "kxar_ai_blk",
    "kxar_ai_dmg",
    "kxar_ai_end",
    "kxar_ai_per",
    "kxar_cleanup",
    "kxar_join",
    "kxar_dlgack",
    *DIRECTOR_POWER_SCRIPT_RESREFS,
)
ENCOUNTER_TRIGGER_POSITION = (-5.625, -22.25, 9.929297798)
ENCOUNTER_TRIGGER_GEOMETRY = (
    (-13.375, -5.25, 0.079067814),
    (9.625, -5.25, 0.282852481),
    (9.625, 5.25, 0.042318766),
    (-5.875, 5.25, -0.334007284),
)
# Proximity polling is intentionally disabled. The authored UTT is the primary
# entry path and Xaria's conventional ScriptDialogue event is the retry path.
ENCOUNTER_PROXIMITY_METRES = 0.0
ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES = 8.0

BUNDLED_ENCOUNTER_RESOURCES = (
    {"resref": PRODUCTION_XARIA_TEMPLATE, "type": "utc", "purpose": "Accepted production cinematic Xaria template."},
    {"resref": PRODUCTION_WRAID_TEMPLATES[0], "type": "utc", "purpose": "Accepted production Miststep Nydak target."},
    {"resref": PRODUCTION_WRAID_TEMPLATES[1], "type": "utc", "purpose": "Accepted production Ichor Lightning Nydak target."},
    {"resref": PRODUCTION_WRAID_TEMPLATES[2], "type": "utc", "purpose": "Accepted production Ichor Drain Nydak target."},
    {
        "resref": PRODUCTION_TRIGGER_TEMPLATE,
        "type": "utt",
        "purpose": "Accepted production encounter trigger.",
    },
    {
        "resref": PRODUCTION_DIRECTOR_TEMPLATE,
        "type": "utp",
        "purpose": "Accepted production cutscene-camera director.",
    },
    {
        "resref": PRODUCTION_TIMELINE_TEMPLATE,
        "type": "utp",
        "purpose": "Accepted production combat timeline owner.",
    },
    {
        "resref": PRODUCTION_CAMERA_DIALOGUE,
        "type": "dlg",
        "purpose": "Accepted production three-camera combat controller.",
    },
    {"resref": PRODUCTION_BLADE, "type": "uti", "purpose": "Accepted Xaria's Blade item."},
    *(
        {
            "resref": resref,
            "type": "uts",
            "purpose": "Accepted production positional effect-sound anchor.",
        }
        for resref, _tag in PRODUCTION_SOUND_OBJECTS
    ),
    *(
        {
            "resref": resref,
            "type": "ncs",
            "purpose": "Accepted compiled production encounter script.",
        }
        for resref in PRODUCTION_ENCOUNTER_NCS
    ),
)

def _parse_streamvoice_resref(resref: str) -> tuple[str, str, str]:
    match = re.fullmatch(
        r"(?P<voice_id>[0-9]{3})(?P<speaker>[A-Za-z0-9_]+)"
        r"(?P<suffix>[0-9]{3})",
        resref,
    )
    if (
        match is None
        or len(resref) > 16
        or int(match.group("voice_id")) <= 0
    ):
        raise RuntimeError(
            "Xaria VO ResRef cannot be resolved by the retail StreamVoice "
            f"loader: {resref}"
        )
    return (
        match.group("voice_id"),
        match.group("speaker").casefold(),
        match.group("suffix"),
    )


def _streamvoice_relative_path(resref: str) -> str:
    voice_id, speaker, _suffix = _parse_streamvoice_resref(resref)
    return f"StreamVoice/{voice_id}/{speaker}/{resref}.wav"


PRODUCTION_SCRIPT_RESREFS = (
    "cxar_post",
    "kxar_dlgack",
    "cxar_wait",
    "cxar_party",
    "cxar_decide",
    "cxar_atton_m",
    "cxar_atton_f",
    "cxar_kreia",
    "cxar_react",
    "cxar_pfirst",
    "cxar_pgrove",
    "cxar_phawk",
    "cxar_pother",
    "cxar_d_ready",
    "cxar_d_busy",
    "cxar_q_name",
    "cxar_q_ichor",
    "cxar_q_recog",
    "cxar_q_war",
    "cxar_q_sphere",
    "cxar_l_veil",
    "cxar_l_heal",
    "cxar_l_drain",
    "cxar_l_mist",
    "cxar_l_light",
    "cxar_l_raise",
    "kxar_join",
    "kxar_cleanup",
    "kxar_wait",
    "kxar_q_name",
    "kxar_q_ichor",
    "kxar_q_recog",
    "kxar_q_war",
    "kxar_q_sphere",
    "kxar_spawn",
    "kxar_d_begin",
    "kxar_d_clean",
    "kxar_d_tick",
    "kxar_d_queue",
    "kxar_d_veil",
    "kxar_d_heal",
    "kxar_d_drain",
    "kxar_d_mist",
    "kxar_d_light",
    "kxar_d_raise",
    "kxar_l_veil",
    "kxar_l_iheal",
    "kxar_l_idrain",
    "kxar_l_mist",
    "kxar_l_ilight",
    "kxar_l_raise",
    "kxar_talkup",
    "kxar_rel_good",
    "kxar_rel_soft",
)

PRODUCTION_VOICE_SOURCE_TO_RUNTIME = {
    "xv_intro": "997xaria001",
    "xv_name": "997xaria002",
    "xv_ichor": "997xaria003",
    "xv_recog": "997xaria004",
    "xv_war": "997xaria005",
    "xv_sphere": "997xaria006",
    "xv_join": "997xaria007",
    "xv_leave": "997xaria008",
    "xv_return": "997xaria009",
    "xv_lhub": "997xaria010",
    "xv_lveil": "997xaria011",
    "xv_lheal": "997xaria012",
    "xv_ldrain": "997xaria013",
    "xv_lmist": "997xaria014",
    "xv_llight": "997xaria015",
    "xv_lraise": "997xaria016",
    "xv_warn": "997xaria017",
    "xv_hunger": "997xaria018",
    "xv_threat": "997xaria019",
    "xv_stories": "997xaria020",
    "xv_strife": "997xaria021",
    "xv_demand": "997xaria022",
    "xv_soulfound": "997xaria023",
    "xv_hero": "997xaria024",
    "xv_seeker": "997xaria025",
    "xv_soulmouth": "997xaria026",
    "xv_property": "997xaria027",
    "xv_askquick": "997xaria028",
    "xv_nydak": "997xaria029",
    "xv_magick": "997xaria030",
    "xv_withhold": "997xaria031",
    **{
        f"997xaria{index:03d}": f"997xaria{index:03d}"
        for index in range(101, 116)
    },
    "997atton001": "997atton001",
    "997atton002": "997atton002",
    "997kreia001": "997kreia001",
}
PRODUCTION_VOICE_SOURCE_RESREFS = tuple(
    PRODUCTION_VOICE_SOURCE_TO_RUNTIME
)
PRODUCTION_VOICE_RESREFS = tuple(
    PRODUCTION_VOICE_SOURCE_TO_RUNTIME.values()
)

PRODUCTION_VOICE_SHA256 = {
    "xv_intro": "9D8C8E47C4E469F25CCE31AEE05B8320C6B66152EC3DBF959AAFD198ACA35364",
    "xv_name": "E5B2427337EE95E3DA82AEC442C91E6AB4463D6568EEDA63E5F2C15E86EFDA9F",
    "xv_ichor": "1A947E42F241E1611C0BAF447A4F6528C17319BE05E03EE77C3BBCD70957C717",
    "xv_recog": "ED0D0AD58CD1B436C858309070C48D0FC0A5EBBBFD641D9EAC86E32874448187",
    "xv_war": "64DE970866182FD1E39BC4C3401E5B7C233C7B12191CA0540F4821E80C9D4DC0",
    "xv_sphere": "641A797642280E67136E0BE802C236BC5637FE8F0B430EED5A0DE5B5A9292B19",
    "xv_join": "41532C763EF2BF9F45BD715570233B714840399641A9924222081725169D810A",
    "xv_leave": "5D93921CDCD45E83E4C9C7E80E1E70FC7CCC51E7F9C31C488A95791B895FD995",
    "xv_return": "9AEB1ADF4D892D1E5A19849B072F535E7C3F62EAB924F3E4BC1396A7F939C9E1",
    "xv_lhub": "270B9C86D6181A0B7005F065D618B35172D36E8584D294795F9933BD125F7237",
    "xv_lveil": "E86D910616ECB25F893FF29A2B0DB55773F87E6FA37BE0BF527AD799447D56CB",
    "xv_lheal": "671704D20E2975C0A8D2AA29D47EA340AADB4D4263BDA57E9FB4FDECFB07A002",
    "xv_ldrain": "52595EB9627262D6D4EED82F97001282704ABD5A3780CBCEFD14FBA97E9D06B0",
    "xv_lmist": "4F9F80375F5E6ADFBCF457FF95E147A9E9E898989FB320E23019916E13D57367",
    "xv_llight": "2B6417FD9F71C32EC33C5EAE66EA1CE719E1D62B0D787E9456EAAEE5F230916D",
    "xv_lraise": "5C6575F0D2FF78F8C76321E6CBDF2AA80B961737F7AE258A3FD3E26C872AD441",
    "xv_warn": "9609D7611D35041F46F8C25D995B742663D979E8FD55DEA52C75A02CD2022DCF",
    "xv_hunger": "B29CDB74777F2F9EE7E0BAC2687790DE2E57B912AFC7AF7C9B3A7E5A6167B06D",
    "xv_threat": "A0BD68F0DD8C26CD4F269CFB9CFE7208945CEE23EE58A4FF8067C066D4AA3765",
    "xv_stories": "CBFDD6C2477A456A057B452538687515668924D9D35F3D0A81483D9FBC4E3FC6",
    "xv_strife": "0F6574B54B7D9B39842AB85BB53042F5DA5C94D9ED8590CE794E3AFAA2C5E102",
    "xv_demand": "58C6E5A955298A97AA4100B57433E5E1563E9C4F2641680A76DDC85687759EB5",
    "xv_soulfound": "5C046237A8AAF576B9F85F85C154CC72551D6D1A2A3F6293CC86B7AED88EA4F8",
    "xv_hero": "F2ECDBA8DAB4AAB4603902C526FE2E7679D5EC09A5F19D08559C71AC58E6953A",
    "xv_seeker": "E1B59FCE77389B833F72416C6B0F07DC5E8CB79A521D1631702B375290A7C419",
    "xv_soulmouth": "E69435B1479A816CF0391527E35EB95A31AC55FF469BA4DEEF8E0355C31B3C5A",
    "xv_property": "146B93B2775FAB8C68AF596413F5A1F45440AD8C8679F7D5F8CD95DCE24C4ECF",
    "xv_askquick": "FD4205A5BC9C2D7054130FBE624FFA31F78AE75EC8BB15E7DCA14E4E6EDA5751",
    "xv_nydak": "9E8D059DB4DB911E98AB59BF7E62461A584ABA31D3C2F444071EA8070A91799F",
    "xv_magick": "C0A227BB35D4D4F4B3B33CD29CE0D72CE314E0B48C8DA0420E783E1BFE5E9ED7",
    "xv_withhold": "3E5973ADBCEC3CC7A280650169782DBE44B9E8B7B6E2D3D16CB060C5D83BB6A7",
    "997xaria101": "A0FF6CBC93E5CA160993195AD6731554D615CC94A7153E1AFC8B4C39BD3DC33F",
    "997xaria102": "6E9DF40EF6762354C3F6CCC9AADA2A9A9477928E80D07783D1663F13F2BFEAB4",
    "997xaria103": "E0D30868B3D7466F74E0CCB9520D9C72B1989732715F2E4F38490DCC66D85474",
    "997xaria104": "238F1DDDF5986EB70636FA08D9F10FF4938BD30C258220337DA9F9E128B4A253",
    "997xaria105": "D67712C4723B1A45699814A5CDFE272F9F1881A9899C96F4154170BF04E5CCE4",
    "997xaria106": "C26D04E8D950C3E80A60E27E628F43DD8234E371CED959B48FFA565970E1BA11",
    "997xaria107": "A8F3628F6B288EF090043F60EABE2F80C7E36902643CF3160199F325A50A5B65",
    "997xaria108": "7F6296C5D26018EB34401AA7FA736FCE078D4CDA66C36A599F2006EE054797E4",
    "997xaria109": "351830CF6E4D501758B9053F9D66668D59EE16BA0A7DAF2E35B1EEED7C2414CD",
    "997xaria110": "CCA940630A5E60AC9FDFEE6636F5175A317E82E4B6FDA3BD3D329ADCC0527D78",
    "997xaria111": "6C9577663A0BD066DC05DE97DBADFAD27C4EB3C977BD71B42ADB33294E01AA36",
    "997xaria112": "736BC9BAF49161FC3EFA75308617F3D848ED92D45B836D311C4E421CA832F0E6",
    "997xaria113": "40342051DDE7A3F95D25E87111A156840853C038B20E862D0E80590422B75F22",
    "997xaria114": "851EE59C262C915373D0ED964767ED415E0CB786DF63C628DA6A21A1030F4C5F",
    "997xaria115": "D9E03D008A265D1FD0F8F3B3E750ED6577E5D28F02DCBD73A8B90D0B47A82A99",
    "997atton001": "649A9B6E6E99C303B0C5C7C8F08A1F859F427D20641846FFF54477DB689BC166",
    "997atton002": "85537C0998CFDC12ED552C4FCCBD0706B6A8A234C5A292A5642B65F921995AEE",
    "997kreia001": "ACB887D5C24B49BB02237CEFAF82E89AF80CF11E6E96E3B23C121CB5EDF41B46",
}

PRODUCTION_LIP_SHA256 = {
    "xv_intro": "8821C6E9C61B221B0896C0247FCA1125E6725E001313AB8BEE663587847BEEFF",
    "xv_name": "6F0B1FF2C89661CAFA00A28A23320B68F32EC452B5AA1E1705BE0FD45A82BD07",
    "xv_ichor": "37C9B41F790E6C677D1574DF35CC3ECA42A8E9131F3124FC796BE866CB1AA21B",
    "xv_recog": "480C4ED1300E692A7E94178894CFF70C69B279617370353FC44F97B73DD447FF",
    "xv_war": "D5DE4CC59E45B33BCBC880F47D7E62AB97961871A58FD2BFB6758563393C2D13",
    "xv_sphere": "0B6058A73990514D51F8C60E393ECC2B66B21D83A78FA101ED64A4BE48065860",
    "xv_join": "38B46C1F1669762BB7EAC32FBF9CD3202CB3320A95656CB18851B3B8024C3370",
    "xv_leave": "B254804E1FD34B04E9C71EF2256662C3D17DD42403DD4C54CCF8F134DE7B6363",
    "xv_return": "F9E55CAC8B96A45E3D06816C79C9CBFFA060B17CE1DC9AD4B0DBC287ECD65C64",
    "xv_lhub": "43C1728EF41FA8A229B74AFDE2E269BC4D54B6E148AC03D241F64B4081CC1DA5",
    "xv_lveil": "518E44332EC138582808154DABF184FE58D3FEC176F690B28477E6624E518447",
    "xv_lheal": "31744F4833C090D7DC311C6863430002712E08AF8DA80FF9BCC807965B1D09A7",
    "xv_ldrain": "B4483BE2BF6DE8B6936D1D9A5F488B5F73B082424FE78ADECF31A32C62DB1BCB",
    "xv_lmist": "9CC27B5942035940053D2DC419BF52F2FA0D7E4EBF86877030771B2C43A6FE72",
    "xv_llight": "03277F5FC52B79D4F67F4365D3A909B556EB7FA2A99DD5CB42CF5C3996C1AFA3",
    "xv_lraise": "B28958BF8CDD276F9AA2C043B452C537DE29DFA6BBDCE6C917A313289F1FA297",
    "xv_warn": "FE2841522F29DE30A3C166F3EA1E215CED4E1008320F2D3E2FC173A69C9A6A0B",
    "xv_hunger": "C05548D3CB6400FFBD69C3B61F13061F21754662F5A4659476D68012D8B293E1",
    "xv_threat": "E4EC9D57A4D2D2DFD282233F46C799BC5EDCC4B3EC573121F76A742A1FE19FD0",
    "xv_stories": "BD6C305A2B1B68AD05B1B8A8CFE15224E371F4CC64C2A4EB2ADCD13D573AD339",
    "xv_strife": "A2FE2638D947F13A8F66A0EFB98BC6BF631627E62D08C40AF06DC464FD6B51F2",
    "xv_demand": "915D00BAA5D48AEFF9135402C2034E5FE6ED1F4E0E67869C6A93A0DD86891082",
    "xv_soulfound": "8AA7B8F49A0010A993381B48E894CF6402831196ADD55F37D265FE359EEEDEB3",
    "xv_hero": "E9D03702757906086C0427FCCBA6B69B7A237216BCFF5247CDB69EB2296A6592",
    "xv_seeker": "C3A3D47CE953ABA8BF85B3FCB55716493091E076C7F2560A34C51E236782DF99",
    "xv_soulmouth": "D7F23122AA15F333FDDDCC1D804D0D0D8C23F217AF9D1CDD42CCABFAB74FDB7F",
    "xv_property": "DA6C125C6BB6BE2D41B0255635847702746589ACB2C27C321669165549612AF0",
    "xv_askquick": "5F6AAE539DADCF43117272D7E90702F5274BD71CE15C9E284DFBC538F6F098FC",
    "xv_nydak": "7378335A51A74508C2BDF945B9C0B600D7FA3FED180382192EEDB5205D8F4E04",
    "xv_magick": "CBD3EF09519C9300DBADBDC60D938B041F3B44EE77CB127A2469AD1E79152ACD",
    "xv_withhold": "D214885785A1804C43F20FBFB9EE5A9A0022A3C174416F178C8481B0DD894F14",
    "997xaria101": "4AC0B2BDB473FF87B1B5790D674C6B85A9E9A3C2040BFEB5CFBAACEB95C38B23",
    "997xaria102": "EC411548809F77D9160653F1CD9AE93AA33D847AE886217723D1845B73EDE61D",
    "997xaria103": "9BCF3952759C1CBD278710B3337094763A13D695DE21E39BD218448AE6A17B86",
    "997xaria104": "3AEE4041580B8D0146215D34B49DD3FE01AA1F0C77A1ACAB3BEA913AD4FAC15E",
    "997xaria105": "F9E2D5BB2AC02E78A7D20E8C28841D084545785A767FFF61F497AA5BF351856B",
    "997xaria106": "0C19BAA1401E4E4E7F67142F1EAB4D136F708C559F0F6BDA5A85FAB7F046F7FC",
    "997xaria107": "0543C13A1F3C02BAD56B921EE26CAC96D243CBC37543F3D4791855913FDB79C2",
    "997xaria108": "570995AB20F14C1B61BDBC16CD991EFED4161076B771B25B21FE56346225F8F8",
    "997xaria109": "224CCBA8F214C5482D4D74368A9F36F68E1426CCD096DFEBBF4514F85D9CB2EE",
    "997xaria110": "49C53221B03107DE5E910CC95B3960C6BADA4AF2E6DF00BC326055621FDFE811",
    "997xaria111": "3091E8D541F901E2694F6D70E8F7C719866013AA6712DB5A660C3314A9BEA23A",
    "997xaria112": "C5D425D59F7748C68E6C0CA7C481CC8943AD6201A9EAB64D7ED018CDA1E114CD",
    "997xaria113": "56E6BBC96D5F2675B6EF3448C6B4D8887E4A979876D4E9766A5D50F0A074CA28",
    "997xaria114": "13427C0F67AB5030E09130E4E710A765EB58F8F7F5FB12D99F3AF4108513164A",
    "997xaria115": "BF8FA3D6BA2009FEBA4ECE7067ABF3C6C5ED4EF5E4F116C62A108974FFF3CE65",
    "997atton001": "A4D67480F3424C4FAC58590C7001F74AFE76117BE1E847516B7D7015CC6A158B",
    "997atton002": "F2F4AA61CF32012C5F084B087EA8DA68F7ADA83069829D94155ABA4DFBD5AFE3",
    "997kreia001": "15FA2FDF1D812002107CB15C294F02C37EF1BE1366F51E8A44C30FCA52EF5FD9",
}

EXTERNAL_XARIA_FILES = (
    {"name": "p_xariabb.mdl", "purpose": "Xaria animated body model."},
    {"name": "p_xariabb.mdx", "purpose": "Xaria animated body geometry."},
    {"name": "p_xariab1.tga", "purpose": "Xaria body texture."},
    {
        "name": "p_xariah6.mdl",
        "purpose": "Verified Xaria facial-performance head model.",
    },
    {
        "name": "p_xariah6.mdx",
        "purpose": "Verified Xaria facial-performance head geometry.",
    },
    {"name": "p_xaria06.tga", "purpose": "Verified Xaria head texture."},
    {
        "name": "p_xaria06.txi",
        "purpose": "Verified Xaria head texture material metadata.",
    },
    {"name": "po_pxaria.tga", "purpose": "Xaria party and dialogue portrait."},
    {"name": "p_xaria.utc", "purpose": "Production recruitable Xaria template."},
    {"name": "xaria.dlg", "purpose": "Production Xaria recruitment and lesson dialogue."},
    {"name": "xaria_blade.uti", "purpose": "Production Xaria's Blade template."},
    {
        "name": "kxar_mstfx.mdl",
        "purpose": "Location-native green Miststep fire-and-forget plume.",
    },
    {
        "name": "kxar_lhand.mdl",
        "purpose": "Hand-attached Ichor Lightning cast effect.",
    },
    {
        "name": "kxar_dhand.mdl",
        "purpose": "Hand-attached Ichor Drain cast effect.",
    },
    {
        "name": "kxar_lght_dur.mdl",
        "purpose": "Ichor Lightning target impact effect.",
    },
    {
        "name": "kxar_dr_dur.mdl",
        "purpose": "Ichor Drain target impact effect.",
    },
    {
        "name": "kxar_h_imp.mdl",
        "purpose": "Ichor Heal impact effect.",
    },
    {
        "name": "kxar_s_mist.wav",
        "purpose": "Authored Miststep activation sound.",
    },
    {
        "name": "kxar_s_ilight.wav",
        "purpose": "Authored Ichor Lightning cast sound.",
    },
    {
        "name": "kxar_s_idrain.wav",
        "purpose": "Authored Ichor Drain cast sound.",
    },
    {
        "name": "kxar_s_raise.wav",
        "purpose": "Authored Raise Dead cast sound.",
    },
    {
        "name": "kxar_s_veil.wav",
        "purpose": "Authored Veil of the Clanless activation sound.",
    },
    {
        "name": "kxar_s_veild.wav",
        "purpose": "Reversed Veil of the Clanless deactivation sound.",
    },
    *(
        {
            "name": f"{resref}.tga",
            "purpose": "Packaged green-ichor particle texture.",
        }
        for resref in (
            "fx_xar_ichor",
            "fx_xconjur",
            "fx_xdrn1",
            "fx_xdrain1",
            "fx_xdot1",
            "fx_xflare1",
            "fx_xheal",
            "fx_xstr",
            "fx_xmist",
            "fx_xmist1",
        )
    ),
    *(
        {
            "name": f"{resref}.ncs",
            "purpose": "Production Xaria recruitment, lesson, or duel script.",
        }
        for resref in PRODUCTION_SCRIPT_RESREFS
    ),
    *(
        {
            "name": f"{runtime_resref}.lip",
            "sha256": PRODUCTION_LIP_SHA256[source_resref],
            "purpose": (
                "Voice Design 1 lip sync for "
                f"{source_resref} -> {runtime_resref}."
            ),
        }
        for source_resref, runtime_resref
        in PRODUCTION_VOICE_SOURCE_TO_RUNTIME.items()
    ),
)

EXTERNAL_XARIA_STREAMVOICE_FILES = tuple(
    {
        "name": _streamvoice_relative_path(runtime_resref),
        "sha256": PRODUCTION_VOICE_SHA256[source_resref],
        "purpose": (
            "Voice Design 1 retail StreamVoice audio for "
            f"{source_resref} -> {runtime_resref}."
        ),
    }
    for source_resref, runtime_resref
    in PRODUCTION_VOICE_SOURCE_TO_RUNTIME.items()
)

TEASER_VOICE_LOOKUP = {
    "module_voice_id": VOICE_STREAM_ID,
    "module_folder": VOICE_STREAM_ID,
    "dialogue_resref": PRODUCTION_DIALOGUE,
    "speaker_folders": tuple(
        sorted(
            {
                _parse_streamvoice_resref(resref)[1]
                for resref in PRODUCTION_VOICE_RESREFS
            }
        )
    ),
    "source_files": tuple(
        row["name"] for row in EXTERNAL_XARIA_STREAMVOICE_FILES
    ),
    "runtime_files": tuple(
        _streamvoice_relative_path(resref)
        for resref in PRODUCTION_VOICE_RESREFS
    ),
}

ENCOUNTER_DEPENDENCIES = (
    {
        "kind": "base_game",
        "resource": "KOTOR II installed resource library",
        "purpose": "Stock room, supermodel, wraid equipment, textures, and sounds.",
    },
    {
        "kind": "2da_row",
        "resource": "appearance.2da",
        "row": 725,
        "purpose": "KPM_Xaria appearance using p_xariabb and verified head row 199.",
    },
    {
        "kind": "2da_row",
        "resource": "heads.2da",
        "row": 199,
        "purpose": "p_xariah6 head and p_xaria06 texture mapping.",
    },
    {
        "kind": "2da_row",
        "resource": "portraits.2da",
        "row": 64,
        "purpose": "po_pxaria portrait mapping retained by the private UTC clone.",
    },
    {
        "kind": "2da_row",
        "resource": "classes.2da",
        "row": 17,
        "purpose": "Dathomir Witch class used by the private UTC clone.",
    },
    *(
        {
            "kind": "2da_row",
            "resource": "spells.2da",
            "row": row,
            "purpose": "Production Dathomir Witch power row.",
        }
        for row in range(286, 299)
    ),
    *(
        {
            "kind": "2da_row",
            "resource": "visualeffects.2da",
            "row": row,
            "purpose": "Xaria green-ichor attachment and impact VFX row.",
        }
        for row in range(9100, 9104)
    ),
    {
        "kind": "runtime_patch",
        "resource": "XariaPowerRuntime/action-862 hook",
        "purpose": "Executes the three private director spell rows and green ichor effects.",
    },
    {
        "kind": "runtime_patch",
        "resource": "CustomClassExtension",
        "purpose": "Provides Dathomir Witch class/Force-point runtime behavior.",
    },
    {
        "kind": "runtime_patch",
        "resource": "PartySelectionExtensionK2",
        "purpose": "Provides the recruited extended party slot and portrait record.",
    },
    *(
        {
            "kind": "override_file",
            "resource": row["name"],
            "purpose": row["purpose"],
        }
        for row in EXTERNAL_XARIA_FILES
    ),
    *(
        {
            "kind": "streamvoice_file",
            "resource": row["name"],
            "purpose": row["purpose"],
        }
        for row in EXTERNAL_XARIA_STREAMVOICE_FILES
    ),
)

# Visual-only room pieces.  Each is centered and grounded by Terrain Kit; none
# contributes WOK collision, so the stock 402dxna walkmesh remains authoritative.
TERRAIN_DRESSING = (
    {
        "id": "left_foreground_root",
        "asset_id": "vanilla_k2_410dxn02_043",
        "role": "foreground_root",
        "position": (-14.0, -20.0, 9.55),
        "rotation_degrees_z": 24.0,
        "scale": 0.66,
        "zone_id": "hero_clearing",
        "footprint_radius": 1.55,
        "clearance_radius": 0.45,
        "purpose": "Frame the reveal with a heavy root-and-rock silhouette.",
        "rationale": "The root stays nine metres left of the combat origin and outside every power lane.",
    },
    {
        "id": "left_foreground_rocks",
        "asset_id": "vanilla_k2_401dxnd_001",
        "role": "foreground_root",
        "position": (-16.5, -29.5, 10.05),
        "rotation_degrees_z": 132.0,
        "scale": 0.72,
        "zone_id": "hero_clearing",
        "footprint_radius": 1.15,
        "clearance_radius": 0.35,
        "purpose": "Give the wide arrival shot a dark lower-left foreground layer.",
        "rationale": "It sits beyond the switchback edge and leaves the 2.4 metre route envelope clear.",
    },
    {
        "id": "right_foreground_rocks",
        "asset_id": "vanilla_k2_401dxnf_004",
        "role": "foreground_root",
        "position": (12.0, -27.0, 10.09),
        "rotation_degrees_z": 228.0,
        "scale": 0.76,
        "zone_id": "hero_clearing",
        "footprint_radius": 1.20,
        "clearance_radius": 0.35,
        "purpose": "Balance the arrival composition and hide the room edge.",
        "rationale": "It remains behind the Miststep camera and outside the teleport arrival clearance.",
    },
    {
        "id": "left_background_tree",
        "asset_id": "vanilla_k2_402dxna_019",
        "role": "background_tree",
        "position": (-13.0, -12.0, 9.55),
        "rotation_degrees_z": 18.0,
        "scale": 0.72,
        "zone_id": "hero_clearing",
        "footprint_radius": 1.35,
        "clearance_radius": 0.40,
        "purpose": "Build a tall left background silhouette for Xaria close-ups.",
        "rationale": "The trunk is outside the dialogue eyeline and more than eight metres from the caster.",
    },
    {
        "id": "right_background_tree",
        "asset_id": "vanilla_k2_402dxna_021",
        "role": "background_tree",
        "position": (12.0, -10.0, 9.97),
        "rotation_degrees_z": 204.0,
        "scale": 0.70,
        "zone_id": "hero_clearing",
        "footprint_radius": 1.35,
        "clearance_radius": 0.40,
        "purpose": "Close the right side of the forest frame behind the wraids.",
        "rationale": "It is beyond the Lightning and Drain targets, preserving side-profile spell silhouettes.",
    },
    {
        "id": "left_vine_curtain",
        "asset_id": "vanilla_k2_401dxnj_029",
        "role": "midground_vines",
        "position": (-8.5, -8.5, 9.65),
        "rotation_degrees_z": 12.0,
        "scale": 0.74,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.55,
        "clearance_radius": 0.20,
        "purpose": "Layer vertical vines behind the dialogue close-up.",
        "rationale": "The curtain is behind Xaria and left of the route to the exit.",
    },
    {
        "id": "right_vine_curtain",
        "asset_id": "vanilla_k2_401dxnj_030",
        "role": "midground_vines",
        "position": (8.5, -8.5, 10.00),
        "rotation_degrees_z": 192.0,
        "scale": 0.76,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.55,
        "clearance_radius": 0.20,
        "purpose": "Create a second depth layer behind the Drain target.",
        "rationale": "The vines remain north of the target and do not cross the caster-to-target beam.",
    },
    {
        "id": "overhead_canopy",
        "asset_id": "vanilla_k2_402dxna_024",
        "role": "canopy_frame",
        "position": (0.0, -7.5, 15.75),
        "rotation_degrees_z": 31.0,
        "scale": 0.68,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.45,
        "clearance_radius": 0.15,
        "purpose": "Cap the clearing with a dark canopy and a fog-catching upper frame.",
        "rationale": "It is raised above actors and is visual-only, so the exit route stays physically open.",
        "allow_path_overlap": True,
    },
)

ACTOR_MARKERS = (
    {
        "id": "xaria_encounter",
        "template_resref": PRODUCTION_XARIA_TEMPLATE,
        "tag": PRODUCTION_XARIA_TAG,
        "position": (0.0, -18.0, 9.919661),
        "bearing": 0.0,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.60,
        "clearance_radius": 0.70,
        "purpose": "Anchor Xaria at the center of the three-beat power demonstration.",
        "rationale": "Every target has a separate angle and at least four metres of cast space.",
        "landmark": True,
    },
    {
        "id": "wraid_1",
        "template_resref": PRODUCTION_WRAID_TEMPLATES[0],
        "tag": PRODUCTION_WRAID_TAGS[0],
        "position": (5.0, -21.0, 10.013909),
        "bearing": math.pi,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.65,
        "clearance_radius": 0.75,
        "purpose": "Receive Miststep: Ambush and the blade follow-through.",
        "rationale": "The target leaves a clear two-metre pocket behind it for rematerialization.",
    },
    {
        "id": "wraid_2",
        "template_resref": PRODUCTION_WRAID_TEMPLATES[1],
        "tag": PRODUCTION_WRAID_TAGS[1],
        "position": (5.0, -17.0, 9.974767),
        "bearing": math.pi,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.65,
        "clearance_radius": 0.75,
        "purpose": "Receive the unobstructed Ichor Lightning hero cast.",
        "rationale": "The target sits due east of Xaria with no prop or other target crossing the beam.",
    },
    {
        "id": "wraid_3",
        "template_resref": PRODUCTION_WRAID_TEMPLATES[2],
        "tag": PRODUCTION_WRAID_TAGS[2],
        "position": (3.0, -13.0, 9.981651),
        "bearing": -2.2,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.65,
        "clearance_radius": 0.75,
        "purpose": "Receive the rising diagonal Ichor Drain beam.",
        "rationale": "The diagonal separates Drain from Lightning in both staging and camera silhouette.",
    },
    {
        "id": "miststep_arrival",
        "template_resref": "script_marker",
        "tag": "Xaria_Miststep_Arrival",
        "position": (7.0, -23.0, 10.049328),
        "bearing": 0.8,
        "zone_id": "hero_clearing",
        "footprint_radius": 0.40,
        "clearance_radius": 0.30,
        "purpose": "Reserve the rematerialization pocket behind Wraid 1.",
        "rationale": "The pocket is walkable, camera-visible, and outside the other two cast lanes.",
        "landmark": True,
    },
)

CAMERA_MARKERS = (
    {
        "id": "camera_arrival_reveal",
        "camera_id": 110,
        "role": "arrival_reveal",
        "position": (-8.0, -28.0, 10.291661),
        "target": (0.0, -18.0, 10.9),
        "field_of_view": 52.0,
        "height": 2.20,
        "pitch": 82.915163,
        "zone_id": "hero_clearing",
        "purpose": "Reveal Xaria and the three separated wraids as the player rounds the switchback.",
        "rationale": "The lens is behind the trigger, aligned with the arrival route, and clear of foreground rocks.",
    },
    {
        "id": "camera_miststep_ambush",
        "camera_id": 111,
        "role": "miststep_ambush",
        "position": (9.0, -24.0, 10.065993),
        "target": (5.8, -21.8, 11.0),
        "field_of_view": 43.0,
        "height": 1.75,
        "pitch": 78.133148,
        "zone_id": "hero_clearing",
        "purpose": "Profile the green departure/arrival mist and blade follow-through.",
        "rationale": "The camera looks across, not through, Wraids 2 and 3 and preserves the arrival pocket.",
    },
    {
        "id": "camera_ichor_lightning",
        "camera_id": 112,
        "role": "ichor_lightning_hero",
        "position": (9.0, -13.0, 9.986575),
        "target": (5.5, -20.0, 11.0),
        "field_of_view": 58.0,
        "height": 1.85,
        "pitch": 83.8986142652013,
        "zone_id": "hero_clearing",
        "purpose": "Capture Xaria and Wraid 2 in one clean green-lightning profile.",
        "rationale": "The camera sits beyond the target and keeps both endpoints against the dark left background.",
    },
    {
        "id": "camera_ichor_drain",
        "camera_id": 113,
        "role": "ichor_drain_profile",
        "position": (-2.0, -10.0, 10.048286),
        "target": (4.5, -17.5, 11.0),
        "field_of_view": 58.0,
        "height": 1.80,
        "pitch": 85.11468497387739,
        "zone_id": "hero_clearing",
        "purpose": "Show the diagonal green Drain beam with the vine curtain behind the victim.",
        "rationale": "The lens is north-west of the caster and does not overlap either prior beat.",
    },
    {
        "id": "camera_dialogue_closeup",
        "camera_id": 114,
        "role": "dialogue_closeup",
        "position": (7.0, -20.0, 10.009488),
        "target": (0.0, -18.0, 11.45),
        "field_of_view": 31.0,
        "height": 1.70,
        "pitch": 87.958647,
        "zone_id": "hero_clearing",
        "purpose": "Provide an eye-level Xaria close-up for facial animation, emissive eyes, and front hair.",
        "rationale": "A dark root-and-tree background isolates her face without backlighting the hair cards.",
    },
    {
        "id": "camera_dialogue_reverse",
        "camera_id": 115,
        "role": "dialogue_reverse",
        "position": (-3.0, -12.0, 9.883207),
        "target": (-4.2, -18.0, 11.25),
        "field_of_view": 35.0,
        "height": 1.70,
        "pitch": 86.882976,
        "zone_id": "hero_clearing",
        "purpose": "Give the player a matching reverse and test Xaria's speaking animation from the opposite side.",
        "rationale": "The reverse avoids the exit and keeps the right forest wall as a clean background.",
    },
)

SHOWCASE_BEATS = (
    {
        "order": 1,
        "power": "Miststep: Ambush",
        "target_tag": PRODUCTION_WRAID_TAGS[0],
        "arrival": (7.0, -23.0, 10.049328),
        "arrival_clearance_m": 1.75,
        "camera_role": "miststep_ambush",
        "unobstructed_sightline": True,
    },
    {
        "order": 2,
        "power": "Ichor Lightning",
        "target_tag": PRODUCTION_WRAID_TAGS[1],
        "camera_role": "ichor_lightning_hero",
        "unobstructed_sightline": True,
    },
    {
        "order": 3,
        "power": "Ichor Drain",
        "target_tag": PRODUCTION_WRAID_TAGS[2],
        "camera_role": "ichor_drain_profile",
        "unobstructed_sightline": True,
    },
)


def _game_dir() -> Path:
    settings_path = ROOT / "settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        configured = Path(str(settings.get("k2_dir") or ""))
        if (configured / "chitin.key").is_file():
            return configured
    configured = Path(os.environ.get("K2_PATH", ""))
    if (configured / "chitin.key").is_file():
        return configured
    fallback = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Knights of the Old Republic II")
    if (fallback / "chitin.key").is_file():
        return fallback
    raise FileNotFoundError(
        "A KOTOR II installation is required to build the Xaria teaser candidate."
    )


def _configured_candidate_dependency_root() -> Path | None:
    configured = os.environ.get(XARIA_CANDIDATE_ROOT_ENV, "").strip()
    return Path(configured).expanduser().resolve() if configured else None


def _require_candidate_manifest_record(
    path: Path,
    raw: object,
    label: str,
) -> None:
    if not isinstance(raw, dict):
        raise RuntimeError(f"Prepared Xaria stage-manifest is missing {label}.")
    expected_size = raw.get("size")
    expected_hash = str(raw.get("sha256") or "").lower()
    if (
        not isinstance(expected_size, int)
        or expected_size <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
    ):
        raise RuntimeError(
            f"Prepared Xaria stage-manifest has an invalid {label} record."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared Xaria candidate file is missing: {path}"
        )
    data = path.read_bytes()
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_hash:
        raise RuntimeError(
            "Prepared Xaria candidate/stage-manifest hash/size mismatch: "
            f"{label}"
        )


def _validate_prepared_candidate_manifest(root: Path) -> None:
    manifest_path = root.parent / "stage-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Prepared Xaria candidate has no sibling stage-manifest.json: "
            f"{manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Prepared Xaria stage-manifest is unreadable: {manifest_path}"
        ) from exc
    valid_stage_state = (
        (manifest.get("installed") is False and manifest.get("install_state") == "not_requested")
        or (manifest.get("installed") is True and manifest.get("install_state") == "installed")
    ) if isinstance(manifest, dict) else False
    if (
        not isinstance(manifest, dict)
        or manifest.get("operation") != "add_xaria_plcaa_encounter"
        or not valid_stage_state
    ):
        raise RuntimeError(
            "Prepared Xaria stage-manifest must be a verified uninstalled or "
            "installed add_xaria_plcaa_encounter candidate."
        )

    module = manifest.get("module")
    module_record = module.get("candidate") if isinstance(module, dict) else None
    _require_candidate_manifest_record(
        root / "plcaa.mod",
        module_record,
        "module.candidate",
    )

    resources = manifest.get("resources")
    resource_records = (
        resources.get("resources")
        if isinstance(resources, dict)
        else None
    )
    if not isinstance(resource_records, dict):
        raise RuntimeError(
            "Prepared Xaria stage-manifest has no resources.resources audit."
        )
    for row in EXTERNAL_XARIA_FILES:
        name = str(row["name"])
        _require_candidate_manifest_record(
            root / "Override" / name,
            resource_records.get(name),
            f"resources.resources[{name}]",
        )

    voice_audio = manifest.get("voice_audio")
    voice_records = (
        voice_audio.get("files")
        if isinstance(voice_audio, dict)
        else None
    )
    if not isinstance(voice_records, dict):
        raise RuntimeError(
            "Prepared Xaria stage-manifest has no voice_audio.files audit."
        )
    if not all(isinstance(name, str) for name in voice_records):
        raise RuntimeError(
            "Prepared Xaria stage-manifest voice_audio.files keys must be paths."
        )
    actual_voice_names = {
        str(name).replace("\\", "/") for name in voice_records
    }
    expected_voice_names = {
        str(row["name"]).replace("\\", "/")
        for row in EXTERNAL_XARIA_STREAMVOICE_FILES
    }
    if (
        len(actual_voice_names) != len(voice_records)
        or actual_voice_names != expected_voice_names
    ):
        missing = sorted(expected_voice_names - actual_voice_names)
        unexpected = sorted(actual_voice_names - expected_voice_names)
        raise RuntimeError(
            "Prepared Xaria stage-manifest voice_audio.files does not match "
            "the complete production voice contract "
            f"(missing={missing}, unexpected={unexpected})."
        )
    for row in EXTERNAL_XARIA_STREAMVOICE_FILES:
        name = str(row["name"]).replace("\\", "/")
        _require_candidate_manifest_record(
            root / "GameRoot" / Path(name),
            voice_records.get(name),
            f"voice_audio.files[{name}]",
        )


def _dependency_source_roots(
    game_dir: Path,
    candidate_root: Path | None = None,
) -> tuple[Path, Path, str]:
    """Resolve one coherent source for Xaria's loose dependency package."""

    if candidate_root is None:
        game_root = game_dir.resolve()
        return game_root / "Override", game_root, "live_game"

    root = candidate_root.expanduser().resolve()
    override_root = root / "Override"
    game_root = root / "GameRoot"
    missing = tuple(
        path
        for path in (override_root, game_root)
        if not path.is_dir()
    )
    if missing:
        raise FileNotFoundError(
            "The prepared Xaria candidate is incomplete; expected both "
            f"{override_root} and {game_root}."
        )
    _validate_prepared_candidate_manifest(root)
    return override_root, game_root, "prepared_candidate"


PRIVATE_SCRIPT_SOURCES = {
    "xt_start": r"""
void main() {
    object oPC = GetEnteringObject();
    if (!GetIsPC(oPC) || oPC != GetFirstPC()) return;
    ExecuteScript("xt_begin", OBJECT_SELF, 101);
}
""",
    "xt_begin": r"""
void EnsureCurrentSchema(object oXaria) {
    if (GetLocalNumber(oXaria, 24) == 14) return;
    SetLocalNumber(oXaria, 24, 14);
    SetLocalNumber(oXaria, 27, 0);
    SetLocalNumber(oXaria, 26, 0);
    SetLocalNumber(oXaria, 25, 0);
    SetLocalNumber(oXaria, 21, 0);
    SetLocalBoolean(oXaria, 60, FALSE);
    SetLocalBoolean(oXaria, 61, FALSE);
    SetLocalBoolean(oXaria, 62, FALSE);
    SetLocalBoolean(oXaria, 63, FALSE);
    SetLocalNumber(oXaria, 28, 0);
}

void main() {
    object oPC = GetFirstPC();
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    object oWraid3 = GetObjectByTag("XT_Wraid_3", 0);
    object oDirector = GetObjectByTag("XT_Director", 0);
    if (!GetIsObjectValid(oPC) || !GetIsObjectValid(oXaria)
        || !GetIsObjectValid(oWraid1)
        || !GetIsObjectValid(oWraid2) || !GetIsObjectValid(oWraid3)
        || !GetIsObjectValid(oDirector)) {
        SendMessageToPC(oPC, "Xaria teaser encounter resources are incomplete.");
        return;
    }
    EnsureCurrentSchema(oXaria);
    int nState = GetLocalNumber(oXaria, 28);
    if (nState == 2) return;
    if (nState == 1) {
        // Recovery must reacquire the cinematic director before dispatching
        // an unfinished beat. Launching a beat directly here puts combat back
        // under the gameplay camera.
        if (
            GetLocalNumber(oXaria, 21) != 0 ||
            GetIsInConversation(oDirector)
        ) {
            SendMessageToPC(oPC, "Xaria's encounter is already in progress.");
            return;
        }
        AssignCommand(oDirector, ClearAllActions());
        AssignCommand(
            oDirector,
            ActionStartConversation(
                oPC,
                "xt_dlg",
                FALSE,
                CONVERSATION_TYPE_CINEMATIC,
                TRUE,
                "",
                "",
                "",
                "",
                "",
                "",
                FALSE,
                -1,
                -1,
                FALSE
            )
        );
        return;
    }
    int nEntryPath = GetRunScriptVar();
    SetLocalNumber(oXaria, 23, nEntryPath);
    SetLocalNumber(oXaria, 28, 1);
    SetLocalNumber(oXaria, 27, 0);
    SetLocalNumber(oXaria, 26, 0);
    SetLocalNumber(oXaria, 25, 0);
    SetLocalNumber(oXaria, 21, 0);
    SetLocalBoolean(oXaria, 61, FALSE);
    SetLocalBoolean(oXaria, 63, FALSE);
    CancelCombat(oPC);
    CancelCombat(oXaria);
    CancelCombat(oWraid1);
    CancelCombat(oWraid2);
    CancelCombat(oWraid3);
    ChangeToStandardFaction(oXaria, STANDARD_FACTION_FRIENDLY_1);
    ChangeToStandardFaction(oWraid1, STANDARD_FACTION_NEUTRAL);
    ChangeToStandardFaction(oWraid2, STANDARD_FACTION_NEUTRAL);
    ChangeToStandardFaction(oWraid3, STANDARD_FACTION_NEUTRAL);
    SetCommandable(TRUE, oXaria);
    SetCommandable(TRUE, oWraid1);
    SetCommandable(TRUE, oWraid2);
    SetCommandable(TRUE, oWraid3);
    AssignCommand(oXaria, ClearAllActions());
    AssignCommand(oWraid1, ClearAllActions());
    AssignCommand(oWraid2, ClearAllActions());
    AssignCommand(oWraid3, ClearAllActions());
    AssignCommand(oDirector, ClearAllActions());
    SendMessageToPC(
        oPC,
        "Xaria encounter entry " + IntToString(nEntryPath) + " started."
    );
    // Camera 111 must own the timeline before the first combat beat starts.
    // Its node script pauses the conversation, then dispatches the first
    // unfinished beat while the authored camera remains active.
    AssignCommand(
        oDirector,
        ActionStartConversation(
            oPC,
            "xt_dlg",
            FALSE,
            CONVERSATION_TYPE_CINEMATIC,
            TRUE,
            "",
            "",
            "",
            "",
            "",
            "",
            FALSE,
            -1,
            -1,
            FALSE
        )
    );
}
""",
    "xt_click": r"""
void EnsureCurrentSchema(object oXaria) {
    if (GetLocalNumber(oXaria, 24) == 14) return;
    SetLocalNumber(oXaria, 24, 14);
    SetLocalNumber(oXaria, 27, 0);
    SetLocalNumber(oXaria, 26, 0);
    SetLocalNumber(oXaria, 25, 0);
    SetLocalNumber(oXaria, 21, 0);
    SetLocalBoolean(oXaria, 60, FALSE);
    SetLocalBoolean(oXaria, 61, FALSE);
    SetLocalBoolean(oXaria, 62, FALSE);
    SetLocalNumber(oXaria, 28, 0);
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    if (!GetIsObjectValid(oXaria)) return;
    EnsureCurrentSchema(oXaria);
    int nState = GetLocalNumber(oXaria, 28);
    if (nState == 0) {
        ExecuteScript("xt_begin", oXaria, 103);
        return;
    }
    if (nState == 1) {
        object oDirector = GetObjectByTag("XT_Director", 0);
        // Never clear and replay a cast that is still waiting for its
        // deterministic death confirmation. Local Number 21 is 1/2/3 while a
        // beat is active and 11/12/13 while a fixed-camera dispatch is armed.
        if (GetLocalNumber(oXaria, 21) != 0) {
            SendMessageToPC(
                GetFirstPC(),
                "Xaria's encounter is already in progress."
            );
            return;
        }
        if (
            GetIsObjectValid(oDirector) &&
            GetIsInConversation(oDirector) == FALSE
        ) {
            ExecuteScript("xt_begin", oXaria, 103);
        }
        SendMessageToPC(GetFirstPC(), "Xaria's encounter is already in progress.");
        return;
    }
    int nGlobalState = GetGlobalNumber("KPM_XARIA_STATE");
    if (nState == 2 && nGlobalState == 3) {
        object oPC = GetFirstPC();
        if (GetIsObjectValid(oPC)) {
            ExecuteScript("kxar_cleanup", oPC, 0);
        }
        SendMessageToPC(GetFirstPC(), "Xaria is already recruited in this campaign save.");
        return;
    }
    if (nState == 2 && nGlobalState != 3) {
        // The completed three-kill proof is authoritative. A click retries
        // the shared handoff even if the camera DLG missed its OnEnd callback
        // and therefore never promoted the production global to state 2.
        ExecuteScript("xt_post", oXaria, -1);
        return;
    }
}
""",
    "xt_b1": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void ApplyMistAtLocation(location lMist) {
    ApplyEffectAtLocation(
        DURATION_TYPE_INSTANT,
        EffectVisualEffect(9100, FALSE),
        lMist,
        0.0
    );
}

void ApplyAuthoredMiststep(object oXaria, object oTarget) {
    vector vArrival = Vector(7.0, -23.0, 10.049328);
    location lArrival = Location(
        vArrival,
        VectorToAngle(GetPosition(oTarget) - vArrival)
    );
    location lOrigin = GetLocation(oXaria);
    ApplyMistAtLocation(lOrigin);
    AssignCommand(oXaria, JumpToLocation(lArrival));
    DelayCommand(0.20, ApplyMistAtLocation(lArrival));
    SetLocalNumber(oXaria, 26, 1);
}

void ApplyBladeImpact(object oTarget) {
    if (!TargetIsAlive(oTarget)) return;
    AssignCommand(oTarget, ClearAllActions());
    AssignCommand(
        oTarget,
        PlayAnimation(ANIMATION_LOOPING_SPASM, 1.0, 0.65)
    );
}

void FinishFirstTarget(object oXaria, object oTarget) {
    if (
        !GetIsObjectValid(oXaria) ||
        GetLocalNumber(oXaria, 28) != 1 ||
        GetLocalNumber(oXaria, 21) != 1
    ) return;
    if (TargetIsAlive(oTarget)) {
        SetPlotFlag(oTarget, FALSE);
        SetMinOneHP(oTarget, FALSE);
        ApplyEffectToObject(
            DURATION_TYPE_INSTANT,
            EffectDamage(
                1000, DAMAGE_TYPE_UNIVERSAL, DAMAGE_POWER_NORMAL
            ),
            oTarget,
            0.0
        );
    }
    DelayCommand(0.20, ExecuteScript("xt_dead", oXaria, 1));
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    if (!GetIsObjectValid(oXaria) || GetLocalNumber(oXaria, 28) != 1) return;

    // Camera 111 owns only beat one. Its polling node can loop while the
    // reaction/death settles, so arm this delayed dispatch exactly once.
    if (GetRunScriptVar() == 0) {
        int nProof = GetLocalNumber(oXaria, 25);
        if (
            (nProof & 1) == 0 &&
            GetLocalNumber(oXaria, 21) == 0
        ) {
            SetLocalNumber(oXaria, 21, 11);
            DelayCommand(1.25, ExecuteScript("xt_b1", oXaria, 1));
        }
        return;
    }
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    if ((GetLocalNumber(oXaria, 25) & 1) != 0) return;
    if (!TargetIsAlive(oWraid1)) {
        ExecuteScript("xt_dead", oXaria, 1);
        return;
    }
    int nBeatLatch = GetLocalNumber(oXaria, 21);
    if (nBeatLatch != 0 && nBeatLatch != 11) return;
    SetLocalNumber(oXaria, 21, 1);

    SetLocalNumber(oXaria, 26, 0);
    ChangeToStandardFaction(oWraid1, STANDARD_FACTION_HOSTILE_1);
    object oArea = GetArea(oXaria);
    MusicBackgroundStop(oArea);
    DelayCommand(0.10, MusicBattlePlay(oArea));
    AssignCommand(oXaria, ClearAllActions());
    AssignCommand(oWraid1, ClearAllActions());
    SetLocalNumber(oWraid1, 22, 1);
    ExecuteScript("kxar_d_mamb", oXaria, 287);
    if (GetLocalNumber(oXaria, 26) != 1) {
        // The production power deliberately rejects an unsafe random
        // destination. This fixed cinematic has a pre-audited arrival pocket,
        // so use the identical mist effect there rather than deadlocking.
        ApplyAuthoredMiststep(oXaria, oWraid1);
    } else {
        SetLocalNumber(oXaria, 27, 1);
    }
    AssignCommand(
        oXaria,
        ActionAttack(oWraid1, FALSE)
    );
    DelayCommand(1.75, ApplyBladeImpact(oWraid1));
    DelayCommand(2.60, FinishFirstTarget(oXaria, oWraid1));
}
""",
    "xt_b2": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void ClearLingeringIchorEffects(object oXaria, object oTarget) {
    effect eEffect = GetFirstEffect(oTarget);
    int nSafety = 0;
    while (GetIsEffectValid(eEffect) && nSafety < 64) {
        int nType = GetEffectType(eEffect);
        if (
            (nType == EFFECT_TYPE_BEAM || nType == EFFECT_TYPE_POISON) &&
            GetEffectCreator(eEffect) == oXaria
        ) {
            RemoveEffect(oTarget, eEffect);
            eEffect = GetFirstEffect(oTarget);
        } else {
            eEffect = GetNextEffect(oTarget);
        }
        nSafety++;
    }
}

void ApplyLightningImpact(object oXaria, object oTarget) {
    if (
        !GetIsObjectValid(oXaria) ||
        !TargetIsAlive(oTarget) ||
        GetLocalNumber(oXaria, 28) != 1
    ) {
        return;
    }
    ExecuteScript("kxar_d_ilight", oXaria, 290);
    if (GetLocalNumber(oXaria, 26) != 2) {
        // The fake spell cast already plays the custom hand-bound model.
        // Never root the stock beam-duration donor on the victim: its linked
        // endpoints collapse into a bright, stranded ground glyph.
        SetLocalNumber(oXaria, 26, 2);
    } else {
        SetLocalNumber(oXaria, 27, GetLocalNumber(oXaria, 27) | 2);
    }
    AssignCommand(oTarget, ClearAllActions());
    AssignCommand(
        oTarget,
        PlayAnimation(ANIMATION_LOOPING_SPASM, 1.0, 0.65)
    );
}

void FinishSecondTarget(object oXaria, object oTarget) {
    if (
        !GetIsObjectValid(oXaria) ||
        GetLocalNumber(oXaria, 28) != 1 ||
        GetLocalNumber(oXaria, 21) != 2
    ) return;
    if (TargetIsAlive(oTarget)) {
        ClearLingeringIchorEffects(oXaria, oTarget);
        SetPlotFlag(oTarget, FALSE);
        SetMinOneHP(oTarget, FALSE);
        ApplyEffectToObject(
            DURATION_TYPE_INSTANT,
            EffectDamage(
                1000, DAMAGE_TYPE_UNIVERSAL, DAMAGE_POWER_NORMAL
            ),
            oTarget,
            0.0
        );
    }
    DelayCommand(0.20, ExecuteScript("xt_dead", oXaria, 2));
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    if (!GetIsObjectValid(oXaria)) return;
    if (GetLocalNumber(oXaria, 28) != 1) return;
    int nProof = GetLocalNumber(oXaria, 25);
    if (GetRunScriptVar() == 0) {
        if (
            (nProof & 1) != 0 &&
            (nProof & 2) == 0 &&
            GetLocalNumber(oXaria, 21) == 0
        ) {
            SetLocalNumber(oXaria, 21, 12);
            DelayCommand(1.25, ExecuteScript("xt_b2", oXaria, 2));
        }
        return;
    }
    if ((nProof & 1) == 0) {
        ExecuteScript("xt_dead", oXaria, 1);
        return;
    }
    if ((nProof & 2) != 0) return;
    if (!TargetIsAlive(oWraid2)) {
        ExecuteScript("xt_dead", oXaria, 2);
        return;
    }
    int nBeatLatch = GetLocalNumber(oXaria, 21);
    if (nBeatLatch == 2) return;
    if (nBeatLatch != 0 && nBeatLatch != 12) return;
    SetLocalNumber(oXaria, 21, 2);

    SetLocalNumber(oXaria, 26, 0);
    ChangeToStandardFaction(oWraid2, STANDARD_FACTION_HOSTILE_1);
    AssignCommand(oXaria, ClearAllActions());
    AssignCommand(oWraid2, ClearAllActions());
    SetLocalNumber(oWraid2, 22, 2);
    AssignCommand(oXaria, SetFacingPoint(GetPosition(oWraid2)));
    AssignCommand(
        oXaria,
        ActionCastFakeSpellAtObject(
            290,
            oWraid2,
            0
        )
    );
    DelayCommand(0.55, ApplyLightningImpact(oXaria, oWraid2));
    DelayCommand(2.80, FinishSecondTarget(oXaria, oWraid2));
}
""",
    "xt_b3": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void ClearLingeringIchorEffects(object oXaria, object oTarget) {
    effect eEffect = GetFirstEffect(oTarget);
    int nSafety = 0;
    while (GetIsEffectValid(eEffect) && nSafety < 64) {
        int nType = GetEffectType(eEffect);
        if (
            (nType == EFFECT_TYPE_BEAM || nType == EFFECT_TYPE_POISON) &&
            GetEffectCreator(eEffect) == oXaria
        ) {
            RemoveEffect(oTarget, eEffect);
            eEffect = GetFirstEffect(oTarget);
        } else {
            eEffect = GetNextEffect(oTarget);
        }
        nSafety++;
    }
}

void ApplyDrainImpact(object oXaria, object oTarget) {
    if (
        !GetIsObjectValid(oXaria) ||
        !TargetIsAlive(oTarget) ||
        GetLocalNumber(oXaria, 28) != 1
    ) {
        return;
    }
    ExecuteScript("kxar_d_idrain", oXaria, 291);
    if (GetLocalNumber(oXaria, 26) != 3) {
        // Drain uses the same hand-bound cast path. Row 9102 is a duration
        // beam model and is invalid as a victim-root impact.
        SetLocalNumber(oXaria, 26, 3);
    } else {
        SetLocalNumber(oXaria, 27, GetLocalNumber(oXaria, 27) | 4);
    }
    AssignCommand(oTarget, ClearAllActions());
    AssignCommand(
        oTarget,
        PlayAnimation(ANIMATION_LOOPING_SPASM, 1.0, 0.65)
    );
}

void FinishThirdTarget(object oXaria, object oTarget) {
    if (
        !GetIsObjectValid(oXaria) ||
        GetLocalNumber(oXaria, 28) != 1 ||
        GetLocalNumber(oXaria, 21) != 3
    ) return;
    if (TargetIsAlive(oTarget)) {
        ClearLingeringIchorEffects(oXaria, oTarget);
        SetPlotFlag(oTarget, FALSE);
        SetMinOneHP(oTarget, FALSE);
        ApplyEffectToObject(
            DURATION_TYPE_INSTANT,
            EffectDamage(
                1000, DAMAGE_TYPE_UNIVERSAL, DAMAGE_POWER_NORMAL
            ),
            oTarget,
            0.0
        );
    }
    DelayCommand(0.20, ExecuteScript("xt_dead", oXaria, 3));
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    object oWraid3 = GetObjectByTag("XT_Wraid_3", 0);
    if (!GetIsObjectValid(oXaria)) return;
    if (GetLocalNumber(oXaria, 28) != 1) return;
    int nProof = GetLocalNumber(oXaria, 25);
    if (GetRunScriptVar() == 0) {
        if (
            (nProof & 3) == 3 &&
            (nProof & 4) == 0 &&
            GetLocalNumber(oXaria, 21) == 0
        ) {
            SetLocalNumber(oXaria, 21, 13);
            DelayCommand(1.25, ExecuteScript("xt_b3", oXaria, 3));
        }
        return;
    }
    if ((nProof & 3) != 3) {
        if ((nProof & 1) == 0) {
            ExecuteScript("xt_dead", oXaria, 1);
        } else {
            ExecuteScript("xt_dead", oXaria, 2);
        }
        return;
    }
    if ((nProof & 4) != 0) return;
    if (!TargetIsAlive(oWraid3)) {
        ExecuteScript("xt_dead", oXaria, 3);
        return;
    }
    int nBeatLatch = GetLocalNumber(oXaria, 21);
    if (nBeatLatch == 3) return;
    if (nBeatLatch != 0 && nBeatLatch != 13) return;
    SetLocalNumber(oXaria, 21, 3);

    SetLocalNumber(oXaria, 26, 0);
    ChangeToStandardFaction(oWraid3, STANDARD_FACTION_HOSTILE_1);
    AssignCommand(oXaria, ClearAllActions());
    AssignCommand(oWraid3, ClearAllActions());
    SetLocalNumber(oWraid3, 22, 3);
    AssignCommand(oXaria, SetFacingPoint(GetPosition(oWraid3)));
    AssignCommand(
        oXaria,
        ActionCastFakeSpellAtObject(
            291,
            oWraid3,
            0
        )
    );
    DelayCommand(0.55, ApplyDrainImpact(oXaria, oWraid3));
    DelayCommand(3.10, FinishThirdTarget(oXaria, oWraid3));
}
""",
    "xt_dead": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void ForceDeath(object oTarget) {
    if (!TargetIsAlive(oTarget)) return;
    SetPlotFlag(oTarget, FALSE);
    SetMinOneHP(oTarget, FALSE);
    ApplyEffectToObject(
        DURATION_TYPE_INSTANT,
        EffectDamage(
            1000, DAMAGE_TYPE_UNIVERSAL, DAMAGE_POWER_NORMAL
        ),
        oTarget,
        0.0
    );
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    object oWraid3 = GetObjectByTag("XT_Wraid_3", 0);
    if (!GetIsObjectValid(oXaria)) return;
    if (GetLocalNumber(oXaria, 28) != 1) return;

    int nBeat = GetRunScriptVar();
    object oTarget = OBJECT_INVALID;
    int nBit = 0;
    if (nBeat == 1) {
        oTarget = oWraid1;
        nBit = 1;
    } else if (nBeat == 2) {
        oTarget = oWraid2;
        nBit = 2;
    } else if (nBeat == 3) {
        oTarget = oWraid3;
        nBit = 4;
    } else {
        return;
    }

    if (TargetIsAlive(oTarget)) {
        ForceDeath(oTarget);
        DelayCommand(0.25, ExecuteScript("xt_dead", oXaria, nBeat));
        return;
    }

    int nProof = GetLocalNumber(oXaria, 25);
    if ((nProof & nBit) != 0) return;
    nProof = nProof | nBit;
    SetLocalNumber(oXaria, 25, nProof);
    // Releasing this latch lets the next death-gated DLG node arm its own
    // 1.25-second camera preroll. xt_dead never changes cameras or dispatches
    // another beat.
    SetLocalNumber(oXaria, 21, 0);

    if (nProof == 7) {
        SetLocalNumber(oXaria, 28, 2);
        object oTrigger = GetObjectByTag("XT_Intro_Trigger", 0);
        if (GetIsObjectValid(oTrigger)) {
            DestroyObject(oTrigger, 0.0, TRUE, 0.0, TRUE);
        }
        // Camera OnEnd is a useful fast path, not the sole authorization for
        // recruitment. This independent retry was the retail-proven schema-8
        // handoff and survives a missed final-camera callback.
        DelayCommand(0.90, ExecuteScript("xt_post", oXaria, -1));
    }
}
""",
    "xt_c1done": r"""
int StartingConditional() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    if (!GetIsObjectValid(oXaria)) return FALSE;
    return (GetLocalNumber(oXaria, 25) & 1) == 1;
}
""",
    "xt_c2done": r"""
int StartingConditional() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    if (!GetIsObjectValid(oXaria)) return FALSE;
    return (GetLocalNumber(oXaria, 25) & 3) == 3;
}
""",
    "xt_c3done": r"""
int StartingConditional() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    if (!GetIsObjectValid(oXaria)) return FALSE;
    return (GetLocalNumber(oXaria, 25) & 7) == 7;
}
""",
    "xt_camend": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void main() {
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    object oWraid3 = GetObjectByTag("XT_Wraid_3", 0);
    if (
        !GetIsObjectValid(oXaria) ||
        GetLocalNumber(oXaria, 28) != 2 ||
        GetLocalNumber(oXaria, 25) != 7 ||
        GetLocalNumber(oXaria, 21) != 0 ||
        TargetIsAlive(oWraid1) ||
        TargetIsAlive(oWraid2) ||
        TargetIsAlive(oWraid3) ||
        GetLocalBoolean(oXaria, 60) == TRUE
    ) {
        return;
    }
    SetLocalBoolean(oXaria, 60, TRUE);
    ExecuteScript("xt_post", oXaria, -1);
}
""",
    "xt_post": r"""
int TargetIsAlive(object oTarget) {
    return GetIsObjectValid(oTarget) && !GetIsDead(oTarget);
}

void main() {
    object oPC = GetFirstPC();
    object oXaria = GetObjectByTag("XT_Xaria", 0);
    object oWraid1 = GetObjectByTag("XT_Wraid_1", 0);
    object oWraid2 = GetObjectByTag("XT_Wraid_2", 0);
    object oWraid3 = GetObjectByTag("XT_Wraid_3", 0);
    if (!GetIsObjectValid(oXaria)) {
        SendMessageToPC(
            oPC,
            "Xaria encounter handoff rejected: the private Xaria actor is missing."
        );
        return;
    }
    int nState = GetLocalNumber(oXaria, 28);
    if (
        nState != 2 ||
        GetLocalNumber(oXaria, 25) != 7 ||
        GetLocalNumber(oXaria, 21) != 0 ||
        GetLocalNumber(oXaria, 26) != 3 ||
        TargetIsAlive(oWraid1) ||
        TargetIsAlive(oWraid2) ||
        TargetIsAlive(oWraid3)
    ) {
        // Presentation can end early without poisoning the encounter. The
        // area-owned combat chain and heartbeat remain free to finish.
        return;
    }
    object oDirector = GetObjectByTag("XT_Director", 0);
    if (
        GetIsObjectValid(oDirector) &&
        GetIsInConversation(oDirector)
    ) {
        // Let the silent camera owner end naturally. k_oei_endconv is an
        // event wrapper, not an operation that closes the active DLG.
        DelayCommand(0.50, ExecuteScript("xt_post", oXaria, -1));
        return;
    }
    // Telemetry only: reaching this point proves camera ownership has been
    // released. Recovery must never require this boolean up front.
    SetLocalBoolean(oXaria, 60, TRUE);
    int nGlobalState = GetGlobalNumber("KPM_XARIA_STATE");
    if (nGlobalState != 4 && GetLocalBoolean(oXaria, 61) == TRUE) return;
    if (nGlobalState != 4 && GetLocalBoolean(oXaria, 63) == TRUE) {
        if (GetRunScriptVar() != 1) return;
        SetLocalBoolean(oXaria, 63, FALSE);
    }
    SetLocalNumber(oXaria, 27, 0);
    MusicBattleStop(GetArea(oXaria));
    DelayCommand(0.10, MusicBackgroundPlay(GetArea(oXaria)));
    if (nGlobalState == 3) {
        SetLocalBoolean(oXaria, 62, FALSE);
        if (GetIsObjectValid(oPC)) {
            ExecuteScript("kxar_cleanup", oPC, 0);
        }
        SendMessageToPC(
            oPC,
            "Xaria is already recruited in this campaign save."
        );
        return;
    }
    SetLocalBoolean(oXaria, 62, TRUE);
    if (
        !GetIsObjectValid(oPC) ||
        GetIsInConversation(oPC) ||
        GetIsInConversation(oXaria)
    ) {
        if (nGlobalState != 4) {
            DelayCommand(0.50, ExecuteScript("xt_post", oXaria, -1));
        }
        return;
    }
    // Mirror stock K2 combat-to-dialogue teardown before launching the
    // production conversation. The dialogue itself confirms success.
    AssignCommand(oPC, SurrenderToEnemies());
    AssignCommand(oXaria, SurrenderToEnemies());
    AssignCommand(oPC, ClearAllActions());
    AssignCommand(oXaria, ClearAllActions());
    ChangeToStandardFaction(oXaria, STANDARD_FACTION_FRIENDLY_1);
    CancelCombat(oPC);
    CancelCombat(oXaria);
    SetCommandable(TRUE, oXaria);
    if (nGlobalState != 4) {
        SetGlobalNumber("KPM_XARIA_STATE", 2);
        SetLocalBoolean(oXaria, 63, TRUE);
    }
    // The shared Xaria body keys its eye and foot mist emitters from
    // locomotion. Force a stable idle and let its one-second hold outlive
    // every embedded particle before the close dialogue camera begins.
    AssignCommand(
        oXaria,
        PlayAnimation(
            ANIMATION_LOOPING_PAUSE,
            1.0,
            1.0
        )
    );
    AssignCommand(
        oXaria,
        ActionStartConversation(
            oPC,
            "xaria",
            FALSE,
            CONVERSATION_TYPE_CINEMATIC,
            TRUE,
            "",
            "",
            "",
            "",
            "",
            "",
            FALSE,
            -1,
            -1,
            FALSE
        )
    );
    if (nGlobalState != 4) {
        // ActionStartConversation can fail silently while the engine finishes
        // tearing down the camera DLG. kxar_dlgack sets local 61 only after the
        // production intro really begins, so retry until that acknowledgement.
        DelayCommand(3.00, ExecuteScript("xt_post", oXaria, 1));
    }
}
""",
    "xt_enddlg": r"""
void main() {
    ExecuteScript("k_oei_endconv", OBJECT_SELF, -1);
    if (
        GetGlobalNumber("KPM_XARIA_STATE") != 3 ||
        GetTag(OBJECT_SELF) != "XT_Xaria"
    ) {
        return;
    }

    int iSlot = GetGlobalNumber("KPM_XARIA_SLOT");
    object oPC = GetFirstPC();
    if (
        GetIsObjectValid(oPC) &&
        iSlot >= 12 &&
        iSlot <= 19
    ) {
        AssignCommand(
            oPC,
            DelayCommand(
                0.2,
                ShowPartySelectionGUI(
                    "k_pend_reset", iSlot, 0xFFFFFFFF, FALSE
                )
            )
        );
        AssignCommand(
            oPC,
            DelayCommand(
                0.05,
                ExecuteScript("kxar_cleanup", oPC, 0)
            )
        );
    }
}
""",
    "xt_cleanup": r"""
void main() {
    object oPC = GetFirstPC();
    if (GetIsObjectValid(oPC) == FALSE) return;
    ExecuteScript("kxar_cleanup", oPC, GetRunScriptVar());
}
""",
    "xt_hb": r"""
void EnsureCurrentSchema(object oXaria) {
    if (GetLocalNumber(oXaria, 24) == 14) return;
    SetLocalNumber(oXaria, 24, 14);
    SetLocalNumber(oXaria, 27, 0);
    SetLocalNumber(oXaria, 26, 0);
    SetLocalNumber(oXaria, 25, 0);
    SetLocalNumber(oXaria, 21, 0);
    SetLocalBoolean(oXaria, 60, FALSE);
    SetLocalBoolean(oXaria, 61, FALSE);
    SetLocalBoolean(oXaria, 62, FALSE);
    SetLocalNumber(oXaria, 28, 0);
}

void main() {
    EnsureCurrentSchema(OBJECT_SELF);
    int nState = GetLocalNumber(OBJECT_SELF, 28);
    if (nState == 1) {
        // Recovery must reacquire the director's cinematic conversation.
        // Directly launching a beat here would return combat to gameplay view.
        if (GetLocalNumber(OBJECT_SELF, 21) != 0) return;
        object oDirector = GetObjectByTag("XT_Director", 0);
        if (
            GetIsObjectValid(oDirector) &&
            GetIsInConversation(oDirector) == FALSE
        ) {
            ExecuteScript("xt_begin", OBJECT_SELF, 102);
        }
        return;
    }
    if (
        nState == 2 &&
        GetGlobalNumber("KPM_XARIA_STATE") != 4 &&
        GetLocalBoolean(OBJECT_SELF, 61) == FALSE
    ) {
        ExecuteScript("xt_post", OBJECT_SELF, -1);
        return;
    }
    if (
        nState == 2 &&
        GetGlobalNumber("KPM_XARIA_STATE") == 3
    ) {
        ExecuteScript("kxar_cleanup", GetFirstPC(), 0);
        return;
    }
}
""",
}


def _private_dialogue_bytes() -> bytes:
    """Build a cyclic three-camera director gated by confirmed actor deaths."""

    from pykotor.common.language import LocalizedString
    from pykotor.common.misc import Game, ResRef
    from pykotor.resource.generics.dlg import (
        DLG,
        DLGEntry,
        DLGLink,
        DLGReply,
        bytes_dlg,
    )

    dlg = DLG()
    dlg.skippable = False
    dlg.on_end = ResRef("xt_camend")
    dlg.on_abort = ResRef("xt_camend")
    dlg.comment = (
        "Camera-owned xartease combat timeline. Cameras 111, 112, and 113 "
        "poll the private proof mask. Each camera dispatches only its own "
        "idempotent beat; the terminal proof-7 reply ends naturally and "
        "xt_camend performs the guarded production-dialogue handoff."
    )
    dlg.next_node_id = 9

    entries: list[DLGEntry] = []
    for index, (camera_id, beat_script) in enumerate(
        ((111, "xt_b1"), (112, "xt_b2"), (113, "xt_b3"))
    ):
        entry = DLGEntry()
        entry.list_index = index
        entry.node_id = index
        entry.plot_index = -1
        entry.plot_xp_percentage = 0.0
        entry.camera_angle = 6
        entry.camera_id = camera_id
        entry.script1 = ResRef(beat_script)
        entry.delay = PRIVATE_CAMERA_POLL_SECONDS
        entry.text = LocalizedString.from_invalid()
        entry.sound_exists = 0
        entry.unskippable = True
        entries.append(entry)

    replies: list[DLGReply] = []
    for index in range(6):
        reply = DLGReply()
        reply.list_index = index
        reply.node_id = 3 + index
        reply.plot_index = -1
        reply.plot_xp_percentage = 0.0
        reply.text = LocalizedString.from_invalid()
        reply.sound_exists = 0
        reply.unskippable = True
        replies.append(reply)

    for stage, condition in enumerate(
        ("xt_c1done", "xt_c2done", "xt_c3done")
    ):
        advance_reply = replies[stage * 2]
        hold_reply = replies[stage * 2 + 1]

        # Link order is deliberate: take the proven-death advance first, then
        # the negated polling loop. Camera 113's proof-7 advance is terminal.
        advance_link = DLGLink(advance_reply, 0)
        advance_link.active1 = ResRef(condition)
        entries[stage].links.append(advance_link)

        hold_link = DLGLink(hold_reply, 1)
        hold_link.active1 = ResRef(condition)
        hold_link.active1_not = True
        entries[stage].links.append(hold_link)

        loop_back = DLGLink(entries[stage], 0)
        loop_back.is_child = True
        hold_reply.links.append(loop_back)
        if stage < 2:
            advance_reply.links.append(DLGLink(entries[stage + 1], 0))

    dlg.starters.append(DLGLink(entries[0], 0))
    return bytes(bytes_dlg(dlg, Game.K2))


def _require_private_faction_contract(factions: dict[str, int]) -> None:
    expected = {
        PRIVATE_XARIA_TEMPLATE: PRIVATE_XARIA_FACTION_ID,
        **{
            resref: PRIVATE_WRAID_FACTION_ID
            for resref in PRIVATE_WRAID_TEMPLATES
        },
    }
    actual = {resref: factions.get(resref) for resref in expected}
    if actual != expected:
        raise RuntimeError(
            "Private Xaria teaser factions must serialize as Friendly_1 with "
            f"neutral pre-cinematic targets: expected {expected}, got {actual}."
        )


def _require_private_script_native_action_abi(
    compiled: dict[str, bytes],
) -> None:
    """Reject NCS whose ACTION operands use PyKotor's non-retail stack order.

    KOTOR II's native ACTION wrappers pop formal parameter zero first.  The
    bytecode must therefore push the final formal first and formal zero last.
    The installed PyKotor compiler currently does the opposite for
    multi-argument engine calls.  That makes retail abort the script at the
    first typed VM pop without showing an NWScript error.

    These checks cover all three independent encounter entry paths, the
    central state writer, both conversation launches, and the recruitment
    cleanup/party-selection boundary.
    """

    from pykotor.common import scriptdefs
    from pykotor.resource.formats.ncs import read_ncs

    scripts = {resref: read_ncs(payload) for resref, payload in compiled.items()}

    for resref, ncs in scripts.items():
        for instruction in ncs.instructions:
            if instruction.ins_type.name != "ACTION":
                continue
            routine_id, emitted_count = map(int, instruction.args)
            declared_count = len(scriptdefs.TSL_FUNCTIONS[routine_id].params)
            if emitted_count != declared_count:
                function_name = scriptdefs.TSL_FUNCTIONS[routine_id].name
                raise RuntimeError(
                    f"{resref}.ncs ACTION {routine_id} {function_name} omits "
                    f"default operands ({emitted_count}/{declared_count}); "
                    "the local retail-ABI compiler requires explicit arguments"
                )

    def action_index(resref: str, routine_id: int, occurrence: int = 0) -> int:
        matches = [
            index
            for index, instruction in enumerate(scripts[resref].instructions)
            if instruction.ins_type.name == "ACTION"
            and int(instruction.args[0]) == routine_id
        ]
        if occurrence >= len(matches):
            raise RuntimeError(
                f"{resref}.ncs is missing ACTION {routine_id} "
                f"occurrence {occurrence}"
            )
        return matches[occurrence]

    def require_tail(
        resref: str,
        routine_id: int,
        expected: tuple[tuple[str, list[Any]], ...],
        occurrence: int = 0,
    ) -> None:
        index = action_index(resref, routine_id, occurrence)
        actual = scripts[resref].instructions[index - len(expected) : index]
        actual_view = tuple(
            (instruction.ins_type.name, list(instruction.args))
            for instruction in actual
        )
        if actual_view != expected:
            function_name = scriptdefs.TSL_FUNCTIONS[routine_id].name
            raise RuntimeError(
                f"{resref}.ncs ACTION {routine_id} {function_name} has "
                f"non-retail operands: expected tail {expected!r}, "
                f"got {actual_view!r}"
            )

    def require_any_tail(
        resref: str,
        routine_id: int,
        expected: tuple[tuple[str, list[Any]], ...],
    ) -> None:
        matches = [
            index
            for index, instruction in enumerate(scripts[resref].instructions)
            if instruction.ins_type.name == "ACTION"
            and int(instruction.args[0]) == routine_id
        ]
        for index in matches:
            actual = scripts[resref].instructions[index - len(expected) : index]
            actual_view = tuple(
                (instruction.ins_type.name, list(instruction.args))
                for instruction in actual
            )
            if actual_view == expected:
                return
        function_name = scriptdefs.TSL_FUNCTIONS[routine_id].name
        raise RuntimeError(
            f"{resref}.ncs ACTION {routine_id} {function_name} never has "
            f"the required retail operand tail {expected!r}"
        )

    # Trigger ExecuteScript: nScriptVar, oTarget, then formal-zero sScript.
    require_tail(
        "xt_start",
        8,
        (("CONSTO", [0]), ("CONSTS", ["xt_begin"])),
    )
    # Heartbeat GetLocalNumber: nIndex below formal-zero OBJECT_SELF.
    require_tail(
        "xt_hb",
        681,
        (("CONSTI", [PRIVATE_STATE_LOCAL]), ("CONSTO", [0])),
        occurrence=1,
    )
    # Click GetObjectByTag: nNth below formal-zero tag string.
    require_tail(
        "xt_click",
        200,
        (("CONSTI", [0]), ("CONSTS", [PRIVATE_XARIA_TAG])),
    )
    # Central state write: nValue, nIndex, then formal-zero creature object.
    require_any_tail(
        "xt_begin",
        682,
        (
            ("CONSTI", [1]),
            ("CONSTI", [PRIVATE_STATE_LOCAL]),
            ("CPTOPSP", [-36, 4]),
        ),
    )
    # The conversation action must leave the director target on top.
    require_tail(
        "xt_begin",
        204,
        (("CONSTS", [PRIVATE_DIALOGUE]), ("CPTOPSP", [-84, 4])),
        occurrence=0,
    )
    require_tail(
        "xt_begin",
        204,
        (("CONSTS", [PRIVATE_DIALOGUE]), ("CPTOPSP", [-88, 4])),
        occurrence=1,
    )
    # Camera 111's Script1 pauses the conversation and dispatches beat one.
    # The beat must not be launched from xt_begin before this camera gate.
    require_any_tail(
        "xt_b1",
        8,
        (
            ("CONSTI", [1]),
            ("CPTOPSP", [-12, 4]),
            ("CONSTS", ["xt_b1"]),
        ),
    )
    # The director has paused the conversation before these actor actions.
    # Miststep's blade follow-through uses the same ActionAttack choreography
    # as Jolee; Lightning and Drain use fake spell casts for their casting
    # animations while production director wrappers apply the real effects.
    require_tail(
        "xt_b1",
        37,
        (
            ("CONSTI", [0]),
            ("CPTOPSP", [-16, 4]),
        ),
    )
    require_tail(
        "xt_b2",
        501,
        (
            ("CONSTI", [0]),
            ("CPTOPSP", [-16, 4]),
            ("CONSTI", [290]),
        ),
    )
    require_tail(
        "xt_b3",
        501,
        (
            ("CONSTI", [0]),
            ("CPTOPSP", [-16, 4]),
            ("CONSTI", [291]),
        ),
    )
    require_any_tail(
        "xt_b1",
        8,
        (
            ("CONSTI", [287]),
            ("CPTOPSP", [-20, 4]),
            ("CONSTS", ["kxar_d_mamb"]),
        ),
    )
    require_tail(
        "xt_b2",
        8,
        (
            ("CONSTI", [290]),
            ("CPTOPSP", [-12, 4]),
            ("CONSTS", ["kxar_d_ilight"]),
        ),
    )
    require_tail(
        "xt_b3",
        8,
        (
            ("CONSTI", [291]),
            ("CPTOPSP", [-12, 4]),
            ("CONSTS", ["kxar_d_idrain"]),
        ),
    )
    require_tail(
        "xt_dead",
        241,
        (
            ("CONSTI", [1]),
            ("CONSTF", [0.0]),
            ("CONSTI", [1]),
            ("CONSTF", [0.0]),
            ("CPTOPSP", [-20, 4]),
        ),
    )
    # Production conversation launch on the private Xaria actor.
    require_tail(
        "xt_post",
        204,
        (("CONSTS", ["xaria"]), ("CPTOPSP", [-88, 4])),
    )
    # Party-selection and the shared PC-owned cleanup keep retail order.
    require_tail(
        "xt_enddlg",
        712,
        (
            ("CONSTI", [0]),
            ("CONSTI", [4294967295]),
            ("CPTOPSP", [-16, 4]),
            ("CONSTS", ["k_pend_reset"]),
        ),
    )
    require_tail(
        "xt_enddlg",
        8,
        (
            ("CONSTI", [0]),
            ("CPTOPSP", [-8, 4]),
            ("CONSTS", ["kxar_cleanup"]),
        ),
        occurrence=1,
    )
    require_tail(
        "xt_cleanup",
        8,
        (("CONSTS", ["kxar_cleanup"]),),
    )


def _require_private_script_retail_local_contract() -> None:
    """Reject local storage that retail KOTOR II cannot represent.

    Creature LocalNumbers are byte values stored in slots 12..28 according to
    the game's own ``nwscript.nss``. LocalBooleans use slots 20..63. Invalid
    number slots read as zero in retail, which previously made the heartbeat
    restart the camera dialogue while every combat beat returned early.
    """

    number_call = re.compile(
        r"\b(?P<call>GetLocalNumber|SetLocalNumber)\s*\(\s*"
        r"(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(?P<index>-?\d+)"
    )
    literal_number_write = re.compile(
        r"\bSetLocalNumber\s*\(\s*"
        r"(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(?P<index>-?\d+)\s*,\s*(?P<value>-?\d+)\s*\)"
    )
    boolean_call = re.compile(
        r"\b(?:GetLocalBoolean|SetLocalBoolean)\s*\(\s*"
        r"(?P<object>[A-Za-z_][A-Za-z0-9_]*)\s*,\s*"
        r"(?P<index>-?\d+)"
    )

    failures: list[str] = []
    for resref, source in PRIVATE_SCRIPT_SOURCES.items():
        for match in number_call.finditer(source):
            index = int(match.group("index"))
            if not 12 <= index <= 28:
                failures.append(
                    f"{resref}: {match.group('call')} uses LocalNumber "
                    f"slot {index}, outside retail range 12..28"
                )
        for match in literal_number_write.finditer(source):
            value = int(match.group("value"))
            if not 0 <= value <= 255:
                failures.append(
                    f"{resref}: SetLocalNumber stores {value}, outside "
                    "retail byte range 0..255"
                )
        for match in boolean_call.finditer(source):
            index = int(match.group("index"))
            if not 20 <= index <= 63:
                failures.append(
                    f"{resref}: LocalBoolean slot {index} is outside "
                    "retail range 20..63"
                )

    declared_number_slots = (
        PRIVATE_ACTIVE_BEAT_LOCAL,
        PRIVATE_DIRECTOR_TARGET_LOCAL,
        PRIVATE_ENTRY_PATH_LOCAL,
        PRIVATE_SCHEMA_LOCAL,
        PRIVATE_SEQUENCE_PROOF_LOCAL,
        PRIVATE_POWER_PROOF_LOCAL,
        PRIVATE_WATCHDOG_LOCAL,
        PRIVATE_STATE_LOCAL,
    )
    if len(set(declared_number_slots)) != len(declared_number_slots):
        failures.append("private LocalNumber slot declarations overlap")
    for slot in declared_number_slots:
        if not 12 <= slot <= 28:
            failures.append(
                f"declared private LocalNumber slot {slot} is outside 12..28"
            )
    for value in (
        PRIVATE_SCHEMA,
        0,
        1,
        2,
        12,
        13,
        *PRIVATE_POWER_TOKENS,
    ):
        if not 0 <= value <= 255:
            failures.append(
                f"declared private LocalNumber value {value} is outside 0..255"
            )

    if failures:
        raise RuntimeError(
            "Private Xaria scripts violate retail local storage:\n- "
            + "\n- ".join(failures)
        )


def _compile_private_script_resources() -> tuple[tuple[str, str, bytes], ...]:
    """Compile private scripts with KOTOR II's live native ACTION stack ABI.

    This is intentionally repository-local.  It temporarily adapts the
    installed PyKotor compiler, restores it even on failure, and refuses
    implicit defaults so reversing the paired argument/parameter views cannot
    alter call semantics.
    """

    _require_private_script_retail_local_contract()

    from pykotor.common.misc import Game
    from pykotor.resource.formats.ncs import bytes_ncs, compile_nss
    from pykotor.resource.formats.ncs.compiler.classes import EngineCallExpression

    original_compile = EngineCallExpression.compile

    def compile_with_native_argument_order(self, ncs, root, block):
        if len(self._args) != len(self._function.params):
            raise RuntimeError(
                f"Private teaser call {self._function.name} must pass every "
                f"parameter explicitly "
                f"({len(self._args)}/{len(self._function.params)} present)"
            )

        original_args = self._args
        original_function = self._function
        native_function = copy.copy(original_function)
        native_function.params = list(reversed(original_function.params))
        self._args = list(reversed(original_args))
        self._function = native_function
        try:
            return original_compile(self, ncs, root, block)
        finally:
            self._args = original_args
            self._function = original_function

    compiled: dict[str, bytes] = {}
    EngineCallExpression.compile = compile_with_native_argument_order
    try:
        for resref, source in PRIVATE_SCRIPT_SOURCES.items():
            payload = bytes(bytes_ncs(compile_nss(source, Game.K2)))
            if not payload:
                raise RuntimeError(
                    f"KOTOR II compilation produced an empty {resref}.ncs"
                )
            compiled[resref] = payload
    finally:
        EngineCallExpression.compile = original_compile

    _require_private_script_native_action_abi(compiled)
    return tuple(
        (resref, "ncs", compiled[resref])
        for resref in PRIVATE_SCRIPT_SOURCES
    )


def _director_wrapper_resources(
    dependency_override: Path,
) -> tuple[tuple[str, str, bytes], ...]:
    missing = tuple(
        f"{resref}.ncs"
        for resref in DIRECTOR_POWER_SCRIPT_RESREFS
        if not (dependency_override / f"{resref}.ncs").is_file()
    )
    if missing:
        raise FileNotFoundError(
            "The selected Xaria dependency snapshot is missing its three "
            "compiled director wrappers: "
            + ", ".join(missing)
        )
    resources = tuple(
        (
            resref,
            "ncs",
            (dependency_override / f"{resref}.ncs").read_bytes(),
        )
        for resref in DIRECTOR_POWER_SCRIPT_RESREFS
    )
    if any(not payload.startswith(b"NCS V1.0") for _resref, _restype, payload in resources):
        raise RuntimeError(
            "A selected Xaria director wrapper is not compiled KOTOR NCS."
        )
    return resources


def _private_encounter_resources(
    game_dir: Path,
    *,
    dependency_override: Path | None = None,
) -> tuple[tuple[str, str, bytes], ...]:
    """Create private combat resources with a guarded production handoff."""

    from pykotor.common.misc import Game, ResRef
    from pykotor.extract.capsule import LazyCapsule
    from pykotor.extract.installation import Installation
    from pykotor.resource.generics.utc import bytes_utc, read_utc
    from pykotor.resource.generics.uti import bytes_uti, read_uti
    from pykotor.resource.generics.utp import bytes_utp, read_utp
    from pykotor.resource.generics.utt import bytes_utt, read_utt
    from pykotor.resource.type import ResourceType

    override = (
        dependency_override.resolve()
        if dependency_override is not None
        else game_dir / "Override"
    )
    required_sources = (
        "p_xaria.utc",
        "xar_nydak1.utc",
        "xar_nydak2.utc",
        "xar_nydak3.utc",
        "xaria_blade.uti",
    )
    missing = tuple(name for name in required_sources if not (override / name).is_file())
    if missing:
        raise FileNotFoundError(
            "The Xaria package must be built before xartease so its private templates can be cloned: "
            + ", ".join(missing)
        )

    blank = ResRef.from_blank()
    xaria = read_utc((override / "p_xaria.utc").read_bytes())
    xaria.resref = ResRef(PRIVATE_XARIA_TEMPLATE)
    xaria.tag = PRIVATE_XARIA_TAG
    xaria.faction_id = PRIVATE_XARIA_FACTION_ID
    # A real Conversation resref is required for the engine's normal
    # ScriptDialogue event. xt_click gates that conventional click path by the
    # private encounter state, so the production dialogue cannot bypass combat.
    xaria.conversation = ResRef(PRODUCTION_DIALOGUE)
    xaria.plot = True
    xaria.min1_hp = True
    # Match retail cutscene actors. The invisible director pauses its dialogue
    # before the showcase, which frees Xaria's ordinary action queue for the
    # same combat-animation pattern used by the stock Jolee encounter.
    xaria.interruptable = True
    for field_name in (
        "on_end_dialog",
        "on_blocked",
        "on_notice",
        "on_spell",
        "on_attacked",
        "on_damaged",
        "on_disturbed",
        "on_end_round",
        "on_spawn",
        "on_rested",
        "on_death",
        "on_user_defined",
    ):
        setattr(xaria, field_name, blank)
    xaria.on_dialog = ResRef("xt_click")
    xaria.on_end_dialog = ResRef("xt_enddlg")
    xaria.on_heartbeat = ResRef("xt_hb")
    for item in xaria.equipment.values():
        if str(item.resref).casefold() == "xaria_blade":
            item.resref = ResRef(PRIVATE_BLADE)

    resources: list[tuple[str, str, bytes]] = [
        (PRIVATE_XARIA_TEMPLATE, "utc", bytes(bytes_utc(xaria, Game.K2)))
    ]
    resources.extend(_director_wrapper_resources(override))
    for index, (source_name, resref, tag) in enumerate(
        zip(
            ("xar_nydak1.utc", "xar_nydak2.utc", "xar_nydak3.utc"),
            PRIVATE_WRAID_TEMPLATES,
            PRIVATE_WRAID_TAGS,
        ),
        start=1,
    ):
        wraid = read_utc((override / source_name).read_bytes())
        wraid.resref = ResRef(resref)
        wraid.tag = tag
        wraid.faction_id = PRIVATE_WRAID_FACTION_ID
        wraid.plot = True
        wraid.min1_hp = True
        wraid.interruptable = True
        wraid.current_hp = max(12, int(wraid.current_hp))
        wraid.max_hp = max(wraid.current_hp, int(wraid.max_hp))
        for field_name in (
            "on_end_dialog",
            "on_blocked",
            "on_heartbeat",
            "on_notice",
            "on_spell",
            "on_attacked",
            "on_damaged",
            "on_disturbed",
            "on_end_round",
            "on_dialog",
            "on_spawn",
            "on_rested",
            "on_death",
            "on_user_defined",
        ):
            setattr(wraid, field_name, blank)
        resources.append((resref, "utc", bytes(bytes_utc(wraid, Game.K2))))

    # Clone the proven generic trigger used by the same stock 402DXN room
    # instead of inheriting unrelated flags from the plcaa recruitment fixture.
    stock_trigger_data = LazyCapsule(
        str(game_dir / "Modules" / "402DXN_s.rim")
    ).resource("newgeneric005", ResourceType.UTT)
    if stock_trigger_data is None:
        raise FileNotFoundError(
            "Stock 402DXN trigger newgeneric005.utt is required for xartease."
        )
    trigger = read_utt(bytes(stock_trigger_data))
    trigger.resref = ResRef(PRIVATE_TRIGGER_TEMPLATE)
    trigger.tag = PRIVATE_TRIGGER_TAG
    trigger.on_enter = ResRef("xt_start")
    trigger.on_exit = blank
    trigger.on_heartbeat = blank
    trigger.on_user_defined = blank
    resources.append(
        (PRIVATE_TRIGGER_TEMPLATE, "utt", bytes(bytes_utt(trigger, Game.K2)))
    )

    director_source = Installation(game_dir).resource(
        "plc_invisible", ResourceType.UTP
    )
    if director_source is None:
        raise FileNotFoundError(
            "Stock plc_invisible.utp is required for the Xaria director."
        )
    director = read_utp(director_source.data)
    director.resref = ResRef(PRIVATE_DIRECTOR_TEMPLATE)
    director.tag = PRIVATE_DIRECTOR_TAG
    director.conversation = ResRef(PRIVATE_DIALOGUE)
    director.faction_id = 5
    director.static = True
    director.useable = False
    director.party_interact = False
    for field_name in (
        "on_closed",
        "on_damaged",
        "on_death",
        "on_end_dialog",
        "on_open_failed",
        "on_heartbeat",
        "on_inventory",
        "on_melee_attack",
        "on_force_power",
        "on_open",
        "on_lock",
        "on_unlock",
        "on_used",
        "on_user_defined",
        "on_disarm",
        "on_trap_triggered",
    ):
        setattr(director, field_name, blank)
    resources.append(
        (
            PRIVATE_DIRECTOR_TEMPLATE,
            "utp",
            bytes(bytes_utp(director, Game.K2)),
        )
    )

    blade = read_uti((override / "xaria_blade.uti").read_bytes())
    blade.resref = ResRef(PRIVATE_BLADE)
    blade.tag = "XT_Xarias_Blade"
    resources.append((PRIVATE_BLADE, "uti", bytes(bytes_uti(blade, Game.K2))))
    resources.append((PRIVATE_DIALOGUE, "dlg", _private_dialogue_bytes()))
    resources.extend(_compile_private_script_resources())
    _require_private_faction_contract(
        {
            resref: int(read_utc(payload).faction_id)
            for resref, restype, payload in resources
            if restype == "utc"
        }
    )
    return tuple(resources)


def _production_encounter_resources(
    game_dir: Path,
    *,
    dependency_override: Path | None = None,
) -> tuple[tuple[str, str, bytes], ...]:
    """Copy the accepted PLCaa encounter spine without rewriting its bytes."""

    override = (
        dependency_override.resolve()
        if dependency_override is not None
        else game_dir / "Override"
    )
    resources: list[tuple[str, str, bytes]] = []
    missing: list[str] = []
    for row in BUNDLED_ENCOUNTER_RESOURCES:
        resref = str(row["resref"])
        restype = str(row["type"])
        path = override / f"{resref}.{restype}"
        if not path.is_file():
            missing.append(path.name)
            continue
        payload = path.read_bytes()
        if restype == "ncs" and not payload.startswith(b"NCS V1.0"):
            raise RuntimeError(
                f"Accepted production script is not executable NCS: {path}"
            )
        resources.append((resref, restype, payload))
    if missing:
        raise FileNotFoundError(
            "The selected Xaria production snapshot is missing encounter resources: "
            + ", ".join(missing)
        )
    return tuple(resources)


def _external_dependency_evidence(
    game_dir: Path,
    *,
    candidate_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Hash every custom loose runtime dependency present at build time."""

    override, dependency_game_root, evidence_source = _dependency_source_roots(
        game_dir,
        candidate_root,
    )
    evidence: list[dict[str, Any]] = []
    for runtime_resref in PRODUCTION_VOICE_RESREFS:
        try:
            voice_id, _speaker, _suffix = _parse_streamvoice_resref(
                runtime_resref
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Xaria Voice Design 1 runtime ResRef cannot be resolved by "
                f"the retail StreamVoice loader: {runtime_resref}"
            ) from exc
        if voice_id != VOICE_STREAM_ID:
            raise RuntimeError(
                "Xaria Voice Design 1 runtime ResRef uses the wrong module "
                f"voice namespace: {runtime_resref}"
            )
    for row in EXTERNAL_XARIA_FILES:
        path = override / str(row["name"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Xaria external runtime dependency is missing: {path}"
            )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected_digest = str(row.get("sha256") or "").lower()
        if expected_digest and digest != expected_digest:
            raise RuntimeError(
                f"Xaria Voice Design 1 dependency changed: {path}"
            )
        evidence.append(
            {
                "kind": "override_file",
                "resource": path.name,
                "size": len(data),
                "sha256": digest,
                "verified": True,
                "evidence_source": evidence_source,
            }
        )
    for row in EXTERNAL_XARIA_STREAMVOICE_FILES:
        relative_path = Path(str(row["name"]))
        path = dependency_game_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Xaria external StreamVoice dependency is missing: {path}"
            )
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != str(row["sha256"]).lower():
            raise RuntimeError(
                f"Xaria Voice Design 1 StreamVoice dependency changed: {path}"
            )
        evidence.append(
            {
                "kind": "streamvoice_file",
                "resource": relative_path.as_posix(),
                "size": len(data),
                "sha256": digest,
                "verified": True,
                "evidence_source": evidence_source,
            }
        )
    evidence.extend(
        {
            **dict(row),
            "verified": False,
            "verification_gate": "final transactional Xaria runtime staging manifest",
        }
        for row in ENCOUNTER_DEPENDENCIES
        if row["kind"] not in {"override_file", "streamvoice_file"}
    )
    return tuple(evidence)


def _placement_intent(
    row: dict[str, Any],
    *,
    label: str,
    asset_ref: str,
    allow_path_overlap: bool = False,
) -> Any:
    from src.core.modules.map_studio_spatial_design import SpatialPlacementIntent

    return SpatialPlacementIntent(
        placement_id=str(row["id"]),
        label=label,
        asset_ref=asset_ref,
        position=tuple(float(value) for value in row["position"]),
        bearing=float(row.get("bearing", math.radians(float(row.get("rotation_degrees_z", 0.0))))),
        zone_id=str(row["zone_id"]),
        purpose=str(row["purpose"]),
        rationale=str(row["rationale"]),
        footprint_radius=float(row.get("footprint_radius", 0.15)),
        clearance_radius=float(row.get("clearance_radius", 0.10)),
        landmark=bool(row.get("landmark", False)),
        allow_path_overlap=bool(row.get("allow_path_overlap", allow_path_overlap)),
    )


def build_spatial_plan() -> Any:
    """Return the UI-visible purpose-led layout contract for this candidate."""

    from src.core.modules.map_studio_spatial_design import (
        SpatialDesignPath,
        SpatialDesignPlan,
        SpatialDesignZone,
    )

    terrain = tuple(
        _placement_intent(
            row,
            label=str(row["id"]).replace("_", " ").title(),
            asset_ref=str(row["asset_id"]),
        )
        for row in TERRAIN_DRESSING
    )
    actors = tuple(
        _placement_intent(
            row,
            label=str(row["tag"]).replace("_", " "),
            asset_ref=str(row["template_resref"]),
            allow_path_overlap=True,
        )
        for row in ACTOR_MARKERS
    )
    cameras = tuple(
        _placement_intent(
            {
                **row,
                "footprint_radius": 0.12,
                "clearance_radius": 0.08,
            },
            label=str(row["role"]).replace("_", " ").title(),
            asset_ref=f"git_camera:{row['camera_id']}",
            allow_path_overlap=True,
        )
        for row in CAMERA_MARKERS
    )
    triggers = (
        _placement_intent(
            {
                "id": "encounter_start_trigger",
                "position": ENCOUNTER_TRIGGER_POSITION,
                "bearing": 0.0,
                "zone_id": "hero_clearing",
                "purpose": "Start the showcase whenever the player enters Xaria's clearing.",
                "rationale": "The clearing-wide volume catches both the western switchback and the direct approach while remaining far from module entry.",
                "footprint_radius": 0.20,
                "clearance_radius": 0.10,
            },
            label="Xaria Encounter Start",
            asset_ref=PRODUCTION_TRIGGER_TEMPLATE,
            allow_path_overlap=True,
        ),
        _placement_intent(
            {
                "id": "dxun_exit_trigger",
                "position": EXIT_POINT,
                "bearing": 0.0,
                "zone_id": "exit_threshold",
                "purpose": "Return the tester to stock Dxun after the showcase.",
                "rationale": "The trigger occupies the room's broad north portal and links to stock From_401DXN.",
                "footprint_radius": 0.20,
                "clearance_radius": 0.10,
                "landmark": True,
            },
            label="Return To Dxun",
            asset_ref="newtransition",
            allow_path_overlap=True,
        ),
    )
    return SpatialDesignPlan(
        name="Xaria: Ichor in the Deep Grove",
        design_intent=(
            "Guide the player through a winding Dxun forest reveal into a compact three-beat Xaria power showcase, "
            "then hold a shot-friendly dialogue clearing with an obvious safe exit."
        ),
        grid_size=0.25,
        player_clearance=1.20,
        zones=(
            SpatialDesignZone(
                "arrival_trail",
                "Lower Arrival Trail",
                "Hide the encounter until the player commits to the forest path.",
                (-12.0, -64.0, 1.0, -50.0),
                level_z=2.70,
            ),
            SpatialDesignZone(
                "switchback",
                "Layered Forest Switchback",
                "Build anticipation while proving path continuity over the stock elevation change.",
                (-30.0, -54.0, -4.0, -23.0),
                level_z=6.00,
            ),
            SpatialDesignZone(
                "hero_clearing",
                "Ichor Showcase Clearing",
                "Keep three combat beats, six shot markers, dialogue, hair motion, and emissive effects readable.",
                (-18.0, -31.0, 15.0, -6.0),
                level_z=10.05,
            ),
            SpatialDesignZone(
                "exit_threshold",
                "North Dxun Exit",
                "Offer an unmistakable return to stock Dxun after testing.",
                (-5.0, -7.0, 7.0, 1.0),
                level_z=9.70,
            ),
        ),
        paths=(
            SpatialDesignPath(
                "arrival_to_showcase",
                "Arrival To Showcase",
                "Preserve the exact PIE-proven route from the lower trail to the hero clearing.",
                ROUTE_POINTS,
                width=2.40,
                level_z=10.15,
            ),
            SpatialDesignPath(
                "showcase_to_exit",
                "Showcase To Exit",
                "Leave a direct post-dialogue route from Xaria to the stock-Dxun transition.",
                EXIT_ROUTE_POINTS,
                width=2.40,
                level_z=10.17,
            ),
        ),
        placements=terrain + actors + cameras + triggers,
    )


def _camera_orientation(
    position: Iterable[float], target: Iterable[float]
) -> tuple[float, float, float, float]:
    """Encode a KOTOR GIT camera yaw toward ``target``.

    Odyssey's GFF Orientation stores quaternion components in ``w, x, y, z``
    order.  A static camera faces along its local axis after a separate
    X-pitch, so its horizontal rotation is the world target bearing minus
    ninety degrees.  Stock K1/K2 cameras and reone's GFF/static-camera readers
    both use this contract.
    """

    px, py, _pz = (float(value) for value in position)
    tx, ty, _tz = (float(value) for value in target)
    target_bearing = math.atan2(ty - py, tx - px)
    encoded_yaw = target_bearing - (math.pi * 0.5)
    return (
        math.cos(encoded_yaw * 0.5),
        0.0,
        0.0,
        math.sin(encoded_yaw * 0.5),
    )


def _rectangle(
    center: tuple[float, float, float],
    half_width: float,
    half_height: float,
) -> tuple[tuple[float, float, float], ...]:
    x, y, z = center
    return (
        (x - half_width, y - half_height, z),
        (x + half_width, y - half_height, z),
        (x + half_width, y + half_height, z),
        (x - half_width, y + half_height, z),
    )


def _encounter_trigger_route_coverage() -> float:
    """Return metres of the authored arrival polyline inside the start volume."""

    center_x, center_y, _center_z = ENCOUNTER_TRIGGER_POSITION
    polygon = tuple(
        (center_x + point[0], center_y + point[1])
        for point in ENCOUNTER_TRIGGER_GEOMETRY
    )

    def clipped_length(a: tuple[float, float], b: tuple[float, float]) -> float:
        x0, y0 = a
        dx = b[0] - x0
        dy = b[1] - y0
        enter = 0.0
        leave = 1.0
        for edge_start, edge_end in zip(
            polygon, polygon[1:] + polygon[:1]
        ):
            edge_x = edge_end[0] - edge_start[0]
            edge_y = edge_end[1] - edge_start[1]
            start_side = (
                edge_x * (y0 - edge_start[1])
                - edge_y * (x0 - edge_start[0])
            )
            direction_side = edge_x * dy - edge_y * dx
            if abs(direction_side) <= 1.0e-9:
                if start_side < 0.0:
                    return 0.0
                continue
            ratio = -start_side / direction_side
            if direction_side > 0.0:
                enter = max(enter, ratio)
            else:
                leave = min(leave, ratio)
            if enter > leave:
                return 0.0
        return math.hypot(dx, dy) * max(0.0, leave - enter)

    return sum(
        clipped_length(a, b)
        for a, b in zip(ROUTE_POINTS, ROUTE_POINTS[1:])
    )


def _encounter_trigger_walkmesh_audit(wok: Any) -> tuple[dict[str, Any], ...]:
    """Prove world-space trigger vertices sit on the exported walkmesh floor."""

    find_face_at = getattr(wok, "find_face_at", None)
    get_height_at = getattr(wok, "get_height_at", None)
    bwm_faces = tuple(getattr(wok, "faces", ()) or ())
    if not callable(find_face_at) or not callable(get_height_at):
        from src.core.modules.authored_walkmesh_sampling import (
            walkmesh_face_at_xy,
            walkmesh_floor_z_at_xy,
        )

    rows: list[dict[str, Any]] = []
    for index, local in enumerate(ENCOUNTER_TRIGGER_GEOMETRY, start=1):
        world = tuple(
            ENCOUNTER_TRIGGER_POSITION[axis] + local[axis]
            for axis in range(3)
        )
        if callable(find_face_at) and callable(get_height_at):
            face = find_face_at(world[0], world[1])
            face_index = next(
                (
                    candidate
                    for candidate, item in enumerate(bwm_faces)
                    if item is face
                ),
                -1,
            )
            floor_z = (
                None if face is None else get_height_at(world[0], world[1])
            )
        else:
            face_index = walkmesh_face_at_xy(wok, world[0], world[1])
            floor_z = walkmesh_floor_z_at_xy(
                wok, face_index, world[0], world[1]
            )
        delta = None if floor_z is None else world[2] - floor_z
        rows.append(
            {
                "point": index,
                "local": [round(float(value), 9) for value in local],
                "world": [round(float(value), 9) for value in world],
                "face_index": int(face_index),
                "floor_z": (
                    None if floor_z is None else round(float(floor_z), 9)
                ),
                "z_delta": None if delta is None else round(float(delta), 9),
                "ok": (
                    face_index >= 0
                    and floor_z is not None
                    and abs(float(delta)) <= 1.0e-5
                ),
            }
        )
    return tuple(rows)


def _authored_gameplay_placements() -> Any:
    from src.core.modules.authored_module_objects import (
        AuthoredCameraInstance,
        AuthoredCreatureInstance,
        AuthoredGameplayPlacement,
        AuthoredPlaceableInstance,
        AuthoredSoundInstance,
        AuthoredTriggerInstance,
        ModuleEntryPoint,
    )

    creatures = tuple(
        AuthoredCreatureInstance(
            template_resref=str(row["template_resref"]),
            tag=str(row["tag"]),
            position=tuple(float(value) for value in row["position"]),
            bearing=float(row["bearing"]),
            instance_id=f"xartease_{row['id']}",
        )
        for row in ACTOR_MARKERS
        if row["template_resref"] != "script_marker"
    )
    triggers = (
        AuthoredTriggerInstance(
            template_resref=PRODUCTION_TRIGGER_TEMPLATE,
            tag=PRODUCTION_TRIGGER_TAG,
            position=ENCOUNTER_TRIGGER_POSITION,
            geometry=ENCOUNTER_TRIGGER_GEOMETRY,
            instance_id="xartease_encounter_start",
        ),
        AuthoredTriggerInstance(
            template_resref="newtransition",
            tag="Xaria_Teaser_Exit",
            position=EXIT_POINT,
            geometry=(
                (-1.5, -0.55, 0.0),
                (1.5, -0.55, 0.0),
                (1.5, 0.55, 0.0),
                (-1.5, 0.55, 0.0),
            ),
            linked_to="From_401DXN",
            linked_to_module="402dxn",
            linked_to_flags=2,
            transition_destination=129401,
            instance_id="xartease_dxun_exit",
        ),
    )
    cameras = tuple(
        AuthoredCameraInstance(
            camera_id=int(row["camera_id"]),
            position=tuple(float(value) for value in row["position"]),
            orientation=_camera_orientation(row["position"], row["target"]),
            field_of_view=float(row["field_of_view"]),
            height=float(row["height"]),
            mic_range=18.0,
            pitch=float(row["pitch"]),
            instance_id=f"xartease_{row['role']}",
        )
        for row in CAMERA_MARKERS
    )
    placeables = (
        AuthoredPlaceableInstance(
            template_resref=PRODUCTION_DIRECTOR_TEMPLATE,
            tag=PRODUCTION_DIRECTOR_TAG,
            position=ENCOUNTER_TRIGGER_POSITION,
            instance_id="xartease_invisible_dialogue_director",
        ),
        AuthoredPlaceableInstance(
            template_resref=PRODUCTION_TIMELINE_TEMPLATE,
            tag=PRODUCTION_TIMELINE_TAG,
            position=ENCOUNTER_TRIGGER_POSITION,
            instance_id="xartease_invisible_combat_timeline",
        ),
    )
    sounds = tuple(
        AuthoredSoundInstance(
            template_resref=template_resref,
            tag=tag,
            position=HERO_CLEARING_CENTER,
            instance_id=f"xartease_{template_resref}",
        )
        for template_resref, tag in PRODUCTION_SOUND_OBJECTS
    )
    facing = math.atan2(
        ROUTE_POINTS[1][1] - ROUTE_POINTS[0][1],
        ROUTE_POINTS[1][0] - ROUTE_POINTS[0][0],
    )
    return AuthoredGameplayPlacement(
        entry_point=ModuleEntryPoint(
            area_resref=MODULE_ROOT,
            position=ENTRY_POINT,
            facing=facing,
        ),
        creatures=creatures,
        triggers=triggers,
        sounds=sounds,
        cameras=cameras,
        placeables=placeables,
        metadata={
            "candidate": "xaria_teaser",
            "three_beat_showcase": [dict(row) for row in SHOWCASE_BEATS],
            "camera_roles": [str(row["role"]) for row in CAMERA_MARKERS],
        },
    )


def _paint_exit_transition_surface(authored: Any, playable_room: str) -> Any:
    """Mark the stock north threshold as Odyssey DOOR surface 18.

    The imported room already exposes the north portal, but its four threshold
    triangles retain stock METAL material 10.  A linked module trigger needs a
    DOOR/transition surface in the exported WOK so readiness and retail pathing
    agree on the threshold's purpose.
    """

    rooms = []
    found = False
    for room in authored.rooms:
        if room.normalised_resref() != playable_room:
            rooms.append(room)
            continue
        wok = copy.deepcopy(room.primitive.wok)
        for face_index in (102, 103, 106, 107):
            if face_index >= len(wok.faces):
                raise RuntimeError(
                    f"Stock {SOURCE_ROOM} WOK no longer has transition face {face_index}."
                )
            wok.faces[face_index].surface = 18
        primitive = replace(
            room.primitive,
            wok=wok,
            metadata={
                **dict(room.primitive.metadata),
                "xaria_teaser_transition_face_indices": [102, 103, 106, 107],
                "xaria_teaser_transition_surface_id": 18,
            },
        )
        rooms.append(
            replace(
                room,
                primitive=primitive,
                metadata={
                    **dict(room.metadata),
                    "xaria_teaser_transition_face_indices": [102, 103, 106, 107],
                },
            )
        )
        found = True
    if not found:
        raise RuntimeError(
            f"Playable room {playable_room} was not found for transition-surface painting."
        )
    return replace(authored, rooms=tuple(rooms))


def _restore_playable_room_lightmaps(
    authored: Any,
    playable_room: str,
    resource_manager: Any,
    *,
    source_primitive: Any | None = None,
) -> Any:
    """Restore the untouched stock room's baked lightmap recipe.

    Environment-kit placement deliberately strips source lightmap names and
    UV2 so ordinary users can relight edited kit pieces.  The Xaria teaser uses
    the complete 402dxna render mesh at its identity transform, however, and
    its authored preview lights are not runtime MDL light nodes.  Reattach only
    the donor lightmap name/UV2 channels after validating that every donor and
    target surface still has the same identity and vertex count.  Geometry,
    diffuse UVs, collision, and authored edits remain owned by the target.
    """

    if source_primitive is None:
        if resource_manager is None:
            raise RuntimeError("Restoring stock lightmaps requires KOTOR II resources.")
        from src.core.modules.authored_imported_mesh import (
            build_imported_mesh_primitive_from_stock_model,
        )
        from src.core.modules.map_studio_stock_content_preview import (
            load_stock_kotor_model,
        )

        source_model = load_stock_kotor_model(
            resource_manager,
            SOURCE_ROOM,
            SOURCE_GAME,
        )
        if source_model is None:
            raise RuntimeError(
                f"Could not load stock {SOURCE_ROOM}.mdl/.mdx for lightmap restoration."
            )
        source_primitive = build_imported_mesh_primitive_from_stock_model(
            source_model,
            room_resref=playable_room,
            source_model=SOURCE_ROOM,
            game=SOURCE_GAME,
        )

    source_surfaces = tuple(getattr(source_primitive, "surfaces", ()) or ())
    rooms = []
    found = False
    for room in authored.rooms:
        if room.normalised_resref() != playable_room:
            rooms.append(room)
            continue
        target_surfaces = tuple(getattr(room.primitive, "surfaces", ()) or ())
        if len(target_surfaces) != len(source_surfaces):
            raise RuntimeError(
                "Stock-lightmap donor no longer matches the playable room: "
                f"{len(source_surfaces)} donor surfaces != {len(target_surfaces)} target surfaces."
            )
        restored_surfaces = []
        restored_names: list[str] = []
        for index, (target, source) in enumerate(
            zip(target_surfaces, source_surfaces, strict=True)
        ):
            if str(target.name).casefold() != str(source.name).casefold():
                raise RuntimeError(
                    f"Stock-lightmap surface {index} changed identity: "
                    f"{target.name!r} != {source.name!r}."
                )
            if len(target.vertices) != len(source.vertices):
                raise RuntimeError(
                    f"Stock-lightmap surface {target.name!r} changed vertex count: "
                    f"{len(target.vertices)} != {len(source.vertices)}."
                )
            lightmap = str(getattr(source, "lightmap", "") or "")
            lightmap_uvs = tuple(getattr(source, "uvs_lm", ()) or ())
            if not lightmap:
                restored_surfaces.append(target)
                continue
            if len(lightmap_uvs) != len(target.vertices):
                raise RuntimeError(
                    f"Stock-lightmap surface {target.name!r} has "
                    f"{len(lightmap_uvs)} UV2 rows for {len(target.vertices)} vertices."
                )
            restored_surfaces.append(
                replace(
                    target,
                    lightmap=lightmap,
                    uvs_lm=lightmap_uvs,
                )
            )
            restored_names.append(str(target.name))
        primitive = replace(
            room.primitive,
            surfaces=tuple(restored_surfaces),
            metadata={
                **dict(room.primitive.metadata),
                "source_lightmaps_removed_for_relighting": False,
                "source_lightmaps_restored_for_retail": True,
                "source_lightmap_surface_count": len(restored_names),
                "source_lightmap_surface_names": restored_names,
                "source_lightmap_donor": SOURCE_ROOM,
            },
        )
        rooms.append(
            replace(
                room,
                primitive=primitive,
                metadata={
                    **dict(room.metadata),
                    "source_lightmaps_restored_for_retail": True,
                    "source_lightmap_surface_count": len(restored_names),
                },
            )
        )
        found = True
    if not found:
        raise RuntimeError(
            f"Playable room {playable_room} was not found for stock-lightmap restoration."
        )
    return replace(authored, rooms=tuple(rooms))


def _semantic_resource_manifest(pack_manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows = tuple(dict(row) for row in pack_manifest.get("source", {}).get("resources", ()) or ())
    return tuple(
        {
            "resref": str(row.get("resref") or ""),
            "restype": str(row.get("restype") or ""),
            "size": int(row.get("size") or 0),
            "sha256": str(row.get("sha256") or ""),
            "source": str(row.get("source") or ""),
        }
        for row in sorted(
            rows, key=lambda item: (str(item.get("resref")), str(item.get("restype")))
        )
    )


def _semantic_digest(rows: Iterable[dict[str, Any]]) -> str:
    payload = json.dumps(tuple(rows), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _distance_point_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = dx * dx + dy * dy
    if length <= 1.0e-12:
        return math.dist(point, start)
    amount = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length))
    return math.dist(point, (start[0] + dx * amount, start[1] + dy * amount))


def _showcase_lane_audit() -> tuple[dict[str, Any], ...]:
    xaria = tuple(float(value) for value in ACTOR_MARKERS[0]["position"])
    targets = {
        row["tag"]: tuple(float(value) for value in row["position"])
        for row in ACTOR_MARKERS
        if str(row["tag"]).startswith("Xaria_Wraid_")
    }
    rows: list[dict[str, Any]] = []
    for beat in SHOWCASE_BEATS:
        target = targets[str(beat["target_tag"])]
        clearance_rows = []
        for prop in TERRAIN_DRESSING:
            distance = _distance_point_segment(
                (float(prop["position"][0]), float(prop["position"][1])),
                (xaria[0], xaria[1]),
                (target[0], target[1]),
            )
            required = float(prop["footprint_radius"]) + float(prop["clearance_radius"]) + 0.35
            clearance_rows.append(
                {
                    "placement_id": str(prop["id"]),
                    "distance_m": round(distance, 4),
                    "required_m": round(required, 4),
                    "clear": distance >= required,
                }
            )
        rows.append(
            {
                "power": str(beat["power"]),
                "target_tag": str(beat["target_tag"]),
                "camera_role": str(beat["camera_role"]),
                "clear": all(row["clear"] for row in clearance_rows),
                "prop_clearances": clearance_rows,
            }
        )
    return tuple(rows)


def _run_pie_route(wok: Any) -> dict[str, Any]:
    from src.core.modules.map_studio_pie import MapStudioPIESession

    session = MapStudioPIESession(wok, game="K2", spawn_position=ENTRY_POINT)
    events: list[Any] = []
    legs = (
        ("entry_to_showcase", HERO_CLEARING_CENTER, 2_400),
        ("showcase_to_exit", EXIT_POINT, 1_800),
    )
    results = []
    for label, destination, frame_limit in legs:
        accepted = bool(session.set_destination(destination, run=True))
        for _index in range(frame_limit):
            if not accepted or session.state.destination is None:
                break
            result = session.advance(1.0 / 30.0)
            events.extend(result.events)
        reached = session.state.destination is None
        results.append(
            {
                "leg": label,
                "accepted": accepted,
                "reached": reached,
                "final_position": [round(float(value), 6) for value in session.state.position],
            }
        )
    return {
        "ok": all(row["accepted"] and row["reached"] for row in results),
        "legs": results,
        "event_kinds": sorted({str(event.kind) for event in events}),
    }


def _draw_blueprint(wok: Any, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 980
    image = Image.new("RGB", (width, height), (7, 12, 10))
    draw = ImageDraw.Draw(image, "RGBA")
    vertices = tuple(
        tuple(float(value) for value in vertex[:3]) for vertex in tuple(wok.verts or ())
    )
    min_x, max_x = -40.0, 17.0
    min_y, max_y = -65.5, 1.5
    margin_left, margin_top, margin_right, margin_bottom = 110, 90, 470, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def xy(point: Iterable[float]) -> tuple[float, float]:
        values = tuple(point)
        x = margin_left + (float(values[0]) - min_x) / (max_x - min_x) * plot_w
        y = margin_top + (max_y - float(values[1])) / (max_y - min_y) * plot_h
        return (x, y)

    for face in tuple(wok.faces or ()):
        indices = (
            int(getattr(face, "v1", -1)),
            int(getattr(face, "v2", -1)),
            int(getattr(face, "v3", -1)),
        )
        if len(indices) != 3 or any(index >= len(vertices) for index in indices):
            continue
        surface = int(getattr(face, "surface", 0) or 0)
        color = (30, 69, 52, 210) if surface != 10 else (80, 88, 74, 220)
        draw.polygon(
            [xy(vertices[index]) for index in indices], fill=color, outline=(66, 116, 90, 80)
        )

    plan = build_spatial_plan()
    zone_colors = {
        "arrival_trail": (38, 87, 75, 36),
        "switchback": (51, 104, 77, 32),
        "hero_clearing": (55, 135, 82, 44),
        "exit_threshold": (92, 142, 105, 42),
    }
    for zone in plan.zones:
        x0, y0, x1, y1 = zone.bounds
        ax, ay = xy((x0, y1))
        bx, by = xy((x1, y0))
        draw.rectangle(
            (ax, ay, bx, by),
            fill=zone_colors.get(zone.zone_id, (30, 80, 65, 24)),
            outline=(95, 181, 135, 125),
            width=2,
        )
        draw.text((ax + 8, ay + 6), zone.label, fill=(176, 225, 196, 225))

    route = [xy(point) for point in ROUTE_POINTS]
    exit_route = [xy(point) for point in EXIT_ROUTE_POINTS]
    draw.line(route, fill=(116, 248, 175, 230), width=10, joint="curve")
    draw.line(exit_route, fill=(116, 248, 175, 230), width=10, joint="curve")

    role_colors = {
        "foreground_root": (173, 124, 76, 255),
        "background_tree": (83, 143, 94, 255),
        "midground_vines": (86, 210, 126, 255),
        "canopy_frame": (44, 103, 68, 255),
    }
    for prop in TERRAIN_DRESSING:
        center = xy(prop["position"])
        radius = 10 + int(float(prop["footprint_radius"]) * 4)
        draw.ellipse(
            (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
            fill=role_colors[prop["role"]],
            outline=(225, 238, 215, 180),
            width=2,
        )

    actor_colors = {
        PRODUCTION_XARIA_TAG: (74, 255, 111, 255),
        PRODUCTION_WRAID_TAGS[0]: (245, 174, 86, 255),
        PRODUCTION_WRAID_TAGS[1]: (245, 174, 86, 255),
        PRODUCTION_WRAID_TAGS[2]: (245, 174, 86, 255),
        "Xaria_Miststep_Arrival": (94, 255, 172, 200),
    }
    for actor in ACTOR_MARKERS:
        center = xy(actor["position"])
        draw.ellipse(
            (center[0] - 9, center[1] - 9, center[0] + 9, center[1] + 9),
            fill=actor_colors[actor["tag"]],
            outline=(255, 255, 255, 210),
            width=2,
        )
        draw.text((center[0] + 12, center[1] - 8), str(actor["tag"]), fill=(232, 240, 232, 230))

    xaria_xy = xy(ACTOR_MARKERS[0]["position"])
    target_by_tag = {actor["tag"]: actor for actor in ACTOR_MARKERS}
    beat_colors = ((99, 255, 161, 255), (75, 255, 90, 255), (41, 212, 119, 255))
    for beat, color in zip(SHOWCASE_BEATS, beat_colors):
        target = xy(target_by_tag[beat["target_tag"]]["position"])
        draw.line((xaria_xy, target), fill=color, width=5)

    for camera in CAMERA_MARKERS:
        center = xy(camera["position"])
        target = xy(camera["target"])
        draw.polygon(
            (
                (center[0] - 6, center[1] - 6),
                (center[0] + 8, center[1]),
                (center[0] - 6, center[1] + 6),
            ),
            fill=(80, 186, 255, 220),
        )
        draw.line((center, target), fill=(80, 186, 255, 80), width=2)

    font = ImageFont.load_default()
    draw.text((40, 24), "XARIA: ICHOR IN THE DEEP GROVE", font=font, fill=(196, 255, 213, 255))
    draw.text(
        (40, 46),
        "Ghost Studio spatial blueprint — structural preview, not retail-game evidence",
        font=font,
        fill=(139, 181, 153, 255),
    )
    panel_x = width - 425
    draw.rounded_rectangle(
        (panel_x, 90, width - 30, height - 80),
        radius=18,
        fill=(11, 23, 18, 240),
        outline=(71, 160, 108, 210),
        width=2,
    )
    lines = [
        "Three-beat showcase",
        "",
        "1  Miststep: Ambush",
        "   Wraid 1 + reserved rear pocket",
        "2  Ichor Lightning",
        "   clean east-west hero lane",
        "3  Ichor Drain",
        "   separated rising diagonal",
        "",
        "Shot markers",
        *[f"- {row['role'].replace('_', ' ')}" for row in CAMERA_MARKERS],
        "",
        "Map guarantees",
        "- KOTOR II assets only",
        "- stock 402dxna WOK",
        "- visual props add no collision",
        "- independent xartease module",
        "- return trigger to 402dxn",
    ]
    y = 118
    for line in lines:
        draw.text(
            (panel_x + 24, y), line, font=font, fill=(214, 235, 220, 255) if line else (0, 0, 0, 0)
        )
        y += 27 if line else 14
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _draw_storyboard(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1500, 560
    image = Image.new("RGB", (width, height), (6, 10, 8))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    panels = (
        ("1 · MISTSTEP: AMBUSH", "departure mist → rear pocket → blade", (58, 232, 139, 255)),
        ("2 · ICHOR LIGHTNING", "clean side-profile hero cast", (90, 255, 88, 255)),
        ("3 · ICHOR DRAIN", "rising diagonal beam into vine backdrop", (41, 197, 114, 255)),
    )
    for index, (title, subtitle, color) in enumerate(panels):
        x0 = 30 + index * 490
        x1 = x0 + 460
        draw.rounded_rectangle(
            (x0, 38, x1, 520), radius=18, fill=(13, 25, 19, 255), outline=color, width=3
        )
        draw.text((x0 + 22, 62), title, font=font, fill=(220, 255, 228, 255))
        draw.text((x0 + 22, 88), subtitle, font=font, fill=(155, 203, 169, 255))
        caster = (x0 + 120, 310)
        target = (x0 + 345, 285 if index == 2 else 310)
        draw.ellipse(
            (caster[0] - 25, caster[1] - 55, caster[0] + 25, caster[1] + 55),
            fill=(62, 151, 88, 255),
            outline=(181, 255, 199, 255),
            width=2,
        )
        draw.ellipse(
            (target[0] - 38, target[1] - 48, target[0] + 38, target[1] + 48),
            fill=(125, 83, 51, 255),
            outline=(235, 182, 126, 255),
            width=2,
        )
        if index == 0:
            arrival = (target[0] + 70, target[1] + 38)
            draw.ellipse(
                (arrival[0] - 34, arrival[1] - 34, arrival[0] + 34, arrival[1] + 34),
                fill=(63, 255, 145, 90),
                outline=color,
                width=3,
            )
            draw.line((caster, arrival), fill=color, width=5)
            draw.text(
                (arrival[0] - 47, arrival[1] + 48),
                "1.75 m clear",
                font=font,
                fill=(177, 255, 199, 255),
            )
        else:
            draw.line((caster, target), fill=color, width=8 if index == 1 else 6)
        camera = (x0 + 250, 445)
        draw.polygon(
            (
                (camera[0] - 12, camera[1] - 9),
                (camera[0] + 14, camera[1]),
                (camera[0] - 12, camera[1] + 9),
            ),
            fill=(77, 182, 255, 255),
        )
        draw.text(
            (camera[0] - 45, camera[1] + 22), "shot marker", font=font, fill=(143, 205, 255, 255)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _manual_workflow() -> str:
    terrain_rows = "\n".join(
        f"{index}. Drag `{row['asset_id']}` to `{tuple(row['position'])}`, rotate Z "
        f"`{row['rotation_degrees_z']}°`, scale `{row['scale']}`. Purpose: {row['purpose']}"
        for index, row in enumerate(TERRAIN_DRESSING, 1)
    )
    camera_rows = "\n".join(
        f"{index}. Add Camera `{row['camera_id']}` at `{tuple(row['position'])}`; FOV `{row['field_of_view']}`; "
        f"Height `{row['height']}`; role `{row['role']}`."
        for index, row in enumerate(CAMERA_MARKERS, 1)
    )
    return f"""# Ghost Studio manual workflow — Xaria teaser map

This is the complete UI recreation path for the candidate. The grove uses the connected KOTOR II library and bundles the exact accepted PLCaa production encounter spine. Xaria's models, full recruitment dialogue, 2DA rows, voice files, and runtime patches remain explicit external prerequisites. The MOD is not standalone.

## Create the map

1. Open **Ghost Studio → Map Studio**.
2. Choose **New Project**, game **KOTOR II**, module root `{MODULE_ROOT}`.
3. Connect the installed KOTOR II resource library.
4. Open **Environment Kit**, filter for `402dxn`, and place `{ROOM_PIECE_ID}` at `(0, 0, 0)`.
5. Keep the complete stock room at its identity transform. The current Map
   Studio UI strips environment-kit lightmaps; the deterministic teaser
   builder restores the eight validated `402dxna_lm0` channels after the KMAP
   snapshot. Until a UI preservation control exists, a direct manual export is
   a layout preview rather than a retail-equivalent lit candidate.
6. Set the player start to `{ENTRY_POINT}` and face it along the lower trail.

## Spatial plan

Open the purpose-led spatial-design overlay and create these zones: Lower Arrival Trail, Layered Forest Switchback, Ichor Showcase Clearing, and North Dxun Exit. Add the 2.4 m route through:

`{ROUTE_POINTS}` then `{EXIT_ROUTE_POINTS}`.

Run **Spatial Audit**. The audit must report no blocking issues before export.

## Forest dressing

{terrain_rows}

These pieces are visual-only rooms. Do not generate collision for them; the imported `402dxna` WOK remains the sole navigation authority.

## Lighting and fog

Open **Environment → World Lighting**, choose **Custom**, and enter:

`{json.dumps(WORLD_LIGHTING, sort_keys=True)}`

Add a subtle green hero-clearing point light and a dim cool arrival fill. The game proof must still confirm hair cards and skin remain readable.

## Encounter placements

1. Place production creature `{PRODUCTION_XARIA_TEMPLATE}` / `{PRODUCTION_XARIA_TAG}` at `{ACTOR_MARKERS[0]["position"]}`.
2. Place `{PRODUCTION_WRAID_TEMPLATES[0]}`, `{PRODUCTION_WRAID_TEMPLATES[1]}`, and `{PRODUCTION_WRAID_TEMPLATES[2]}` at their manifest positions.
3. Place `{PRODUCTION_TRIGGER_TEMPLATE}` / `{PRODUCTION_TRIGGER_TAG}` at `{ENCOUNTER_TRIGGER_POSITION}` with
   local geometry `{ENCOUNTER_TRIGGER_GEOMETRY}`. It must span the full final
   corridor, not merely touch a route corner.
4. Place `{PRODUCTION_DIRECTOR_TEMPLATE}` / `{PRODUCTION_DIRECTOR_TAG}` and
   `{PRODUCTION_TIMELINE_TEMPLATE}` / `{PRODUCTION_TIMELINE_TAG}` beside the
   trigger at `{ENCOUNTER_TRIGGER_POSITION}`. The director owns
   `{PRODUCTION_CAMERA_DIALOGUE}`; the separate timeline remains free to run
   the accepted `kxar_show` sequence.
5. Place the three sound anchors `{PRODUCTION_SOUND_OBJECTS}` at
   `{HERO_CLEARING_CENTER}`.
6. Leave the reserved Miststep arrival pocket at `{SHOWCASE_BEATS[0]["arrival"]}` completely clear.
7. Copy the bundled resources byte-for-byte from the accepted production
   candidate. Do not rename tags, clone the UTCs, or recompile the NCS files.
8. Let production `kxar_join`, `kxar_cleanup`, and `kxar_enddlg` own recruitment
   and retirement of the encounter actor.

## Cameras

{camera_rows}

Cameras `111`–`113` are the complete accepted combat sequence. Camera `114`
remains a production-dialogue marker only and is never entered by `xar_cam`.
Production `xaria.dlg` owns its dialogue close-ups and reverse shots. Camera
orientation is GFF `w,x,y,z`, and pitch is stored in degrees.

## Exit

Place trigger `newtransition` at `{EXIT_POINT}` and set:

- LinkedTo: `From_401DXN`
- LinkedToModule: `402dxn`
- LinkedToFlags: `2` (waypoint)
- TransitionDestin: `129401`

In **Walkmesh**, paint threshold faces `102`, `103`, `106`, and `107` as
surface `18` (`DOOR`) so linked-transition readiness and the exported WOK
describe the same threshold.

## Validate, save, and export

1. Generate/inspect the combined WOK.
2. Run PIE from entry to the hero clearing and from the clearing to the exit.
3. Run the spatial audit and placement readback.
4. Save `{MODULE_ROOT}.kmap`.
5. Export a candidate MOD without installing it.
6. Inspect ARE/GIT/IFO/LYT/PTH/VIS/WOK readback and confirm all three creature
   targets, director, timeline, three sound anchors, six correctly encoded
   cameras, the exact accepted DLG graph, and eight stock lightmap channels.

## Remaining retail gates

- Stage the Xaria runtime/assets first, then install through the normal transactional KOTOR patch workflow only while the game is closed.
- `warp {MODULE_ROOT}` and record a visible traversal from entry to clearing to exit.
- Confirm approaching starts combat immediately with no intro dialogue.
- Confirm cameras switch with Miststep, Lightning, and Drain; each beat kills
  its own target before the next camera, then production dialogue begins.
- Confirm a successful join fills the reported extended slot, opens party
  selection, and permanently removes only the old `Xaria_Encounter` origin.
- Repeat with production Xaria already recruited; the encounter may run
  but must never downgrade committed global state `3`.
- Confirm Xaria's hair simulation, front strands, face animation, green eyes, mist footsteps, and dialogue close-ups.
- Confirm the return trigger arrives in stock `402dxn`.
- Treat the included PNGs as structural previews, not retail-game evidence.
"""


def _canonical_kmap(controller: Any, path: Path) -> None:
    from src.core.level.kmap_serializer import KMapSerializer

    controller.project.project_id = "xartease-map-studio-candidate"
    controller.project.created_at = "2026-07-25T00:00:00Z"
    controller.project.modified_at = "2026-07-25T00:00:00Z"
    payload = KMapSerializer.to_dict(controller.project)
    payload["project"]["modified_at"] = "2026-07-25T00:00:00Z"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_once(
    output_dir: Path,
    *,
    candidate_dependency_root: Path | None = None,
) -> dict[str, Any]:
    from pykotor.resource.formats.bwm import read_bwm
    from pykotor.resource.formats.gff import read_gff
    from pykotor.resource.generics.dlg import read_dlg
    from pykotor.resource.generics.utc import read_utc
    from pykotor.resource.generics.utp import read_utp
    from pykotor.resource.generics.uts import read_uts
    from pykotor.resource.generics.utt import read_utt
    from src.core.assets.resource_manager import ResourceManager
    from src.core.modules.authored_module_export import (
        AuthoredModuleExportRequest,
        export_authored_module_project,
    )
    from src.core.modules.authored_module_kmap_bridge import (
        authored_project_from_kmap_payload,
        authored_project_to_kmap_payload,
    )
    from src.core.modules.authored_module_objects import (
        validate_authored_gameplay_placement_against_walkmesh,
    )
    from src.core.modules.authored_module_project import validate_authored_module_project
    from src.core.modules.authored_module_walkmesh import combine_authored_module_walkmesh
    from src.core.modules.map_studio_stock_content_preview import (
        load_kotor_model_from_bytes,
    )
    from src.core.modules.module_editor_controller import ModuleEditorController

    from src.core.modules.map_studio_spatial_design import audit_spatial_design

    game_dir = _game_dir()
    dependency_override, _dependency_game_root, _evidence_source = (
        _dependency_source_roots(game_dir, candidate_dependency_root)
    )
    trigger_route_coverage = _encounter_trigger_route_coverage()
    if trigger_route_coverage < ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES:
        raise RuntimeError(
            "Encounter trigger does not deeply cross the authored approach: "
            f"{trigger_route_coverage:.3f}m < "
            f"{ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES:.3f}m."
        )
    private_resources = _production_encounter_resources(
        game_dir,
        dependency_override=dependency_override,
    )
    external_dependency_evidence = _external_dependency_evidence(
        game_dir,
        candidate_root=candidate_dependency_root,
    )
    resources = ResourceManager()
    if not resources.set_k2_dir(str(game_dir)):
        raise RuntimeError(f"Could not index KOTOR II resources from {game_dir}")

    controller = ModuleEditorController()
    controller.new_project(name=MODULE_ROOT, game="K2", author=AUTHOR)
    playable_room = controller.add_authored_environment_kit_piece(
        piece_id=ROOM_PIECE_ID,
        position=(0.0, 0.0, 0.0),
        resource_manager=resources,
    )
    dressing_rooms = []
    for row in TERRAIN_DRESSING:
        dressing_rooms.append(
            controller.add_authored_terrain_kit_asset(
                asset_id=str(row["asset_id"]),
                position=tuple(float(value) for value in row["position"]),
                rotation_degrees_z=float(row["rotation_degrees_z"]),
                scale=float(row["scale"]),
                target_room_resref=playable_room,
                resource_manager=resources,
            )
        )
    controller.set_authored_world_lighting_settings(dict(WORLD_LIGHTING))
    controller.add_authored_room_light(
        room_resref=playable_room,
        name="Xaria Ichor Key",
        position=(0.0, -17.5, 13.2),
        color=(0.12, 1.0, 0.28),
        radius=13.0,
        intensity=0.85,
    )
    controller.add_authored_room_light(
        room_resref=playable_room,
        name="Lower Grove Fill",
        position=(-14.0, -44.0, 8.0),
        color=(0.16, 0.30, 0.22),
        radius=20.0,
        intensity=0.42,
    )

    authored = authored_project_from_kmap_payload(
        controller.project.extra_sections["authored_module"]
    )
    authored = _paint_exit_transition_surface(authored, playable_room)
    authored = _restore_playable_room_lightmaps(
        authored,
        playable_room,
        resources,
    )
    authored = replace(
        authored,
        metadata=replace(
            authored.metadata,
            display_name="Xaria: Ichor in the Deep Grove",
            tag="Xaria_Teaser",
            description="Independent KOTOR II cinematic test grove for Xaria's witch powers and companion presentation.",
            capability_stage="export_candidate",
            metadata={
                **dict(authored.metadata.metadata),
                "source_game": "K2",
                "source_module": SOURCE_MODULE,
                "source_room": SOURCE_ROOM,
                "legal_asset_policy": "installed_k2_assets_only",
                "voice_over_id": TEASER_VOICE_LOOKUP["module_voice_id"],
            },
        ),
        placements=_authored_gameplay_placements(),
        notes=tuple(authored.notes)
        + (
            "Separate from plcaa; do not replace the recruitment regression fixture.",
            "The exact accepted PLCaa xar_*/kxar_* encounter spine is bundled in this MOD without byte rewriting.",
            "Xaria's full recruitment dialogue, models, global 2DA rows, runtime hooks, and voice files remain external dependencies.",
            "Three-beat order: Miststep Ambush, Ichor Lightning, Ichor Drain.",
            "The accepted production director and timeline objects run xar_cam across cameras 111, 112, and 113, then hand off once to xaria.dlg.",
            "The untouched 402dxna room keeps its eight stock 402dxna_lm0 channels so retail rendering does not collapse into green fog over black geometry.",
            "Visual terrain-kit dressing has zero WOK faces and cannot alter collision.",
        ),
        extra={
            **dict(authored.extra),
            "candidate_id": "xaria_teaser_v1",
            "runtime_dependencies": [dict(row) for row in ENCOUNTER_DEPENDENCIES],
            "bundled_encounter_resources": [
                dict(row) for row in BUNDLED_ENCOUNTER_RESOURCES
            ],
            "production_recruitment_state_access": True,
            "showcase_beats": [dict(row) for row in SHOWCASE_BEATS],
            "camera_markers": [dict(row) for row in CAMERA_MARKERS],
            "retail_game_proof": False,
        },
    )
    controller.project.extra_sections["authored_module"] = authored_project_to_kmap_payload(
        authored
    )
    spatial_audit = controller.set_map_studio_spatial_design(build_spatial_plan())
    if not spatial_audit.ok:
        raise RuntimeError("Spatial design failed: " + "; ".join(spatial_audit.blocking_issues))
    authored = authored_project_from_kmap_payload(
        controller.project.extra_sections["authored_module"]
    )

    project_validation = validate_authored_module_project(authored)
    combined = combine_authored_module_walkmesh(authored)
    if combined.blocking_issues:
        raise RuntimeError("Combined WOK failed: " + "; ".join(combined.blocking_issues))
    placement_validation = validate_authored_gameplay_placement_against_walkmesh(
        authored.placements, combined.wok
    )
    lane_audit = _showcase_lane_audit()
    if not all(row["clear"] for row in lane_audit):
        raise RuntimeError("A forest prop blocks one or more power showcase lanes.")
    pie = _run_pie_route(combined.wok)
    if not pie["ok"]:
        raise RuntimeError(f"PIE traversal failed: {pie}")

    output_dir.mkdir(parents=True, exist_ok=True)
    kmap_path = output_dir / f"{MODULE_ROOT}.kmap"
    _canonical_kmap(controller, kmap_path)
    export_dir = output_dir / "export"
    export = export_authored_module_project(
        AuthoredModuleExportRequest(
            project=authored,
            output_dir=str(export_dir),
            game_root_dir=str(game_dir),
            include_reference_check=False,
            include_wok_check=True,
            include_game_template_check=False,
            strict=True,
            dry_run=False,
            create_backups=False,
            write_loose_resources=True,
            extra_resources=private_resources,
        )
    )
    if not export.ok or export.package_verification is None or not export.package_verification.ok:
        raise RuntimeError(
            "Export/readback failed: "
            + "; ".join(tuple(export.blocking_issues) + tuple(export.warnings[:4]))
        )
    pack_manifest = json.loads(Path(export.manifest_path).read_text(encoding="utf-8"))
    semantic_rows = _semantic_resource_manifest(pack_manifest)
    source_dir = Path(pack_manifest["source"]["resources_dir"])
    trigger_walkmesh_audit = _encounter_trigger_walkmesh_audit(
        read_bwm((source_dir / f"{playable_room}.wok").read_bytes())
    )
    if not all(row["ok"] for row in trigger_walkmesh_audit):
        raise RuntimeError(
            "Encounter trigger has a vertex off the packed walkmesh floor: "
            f"{trigger_walkmesh_audit}"
        )
    packaged_resource_keys = {
        (str(row["resref"]).casefold(), str(row["restype"]).casefold())
        for row in pack_manifest["source"]["resources"]
    }
    expected_private_keys = {
        (str(row["resref"]).casefold(), str(row["type"]).casefold())
        for row in BUNDLED_ENCOUNTER_RESOURCES
    }
    missing_private_keys = expected_private_keys - packaged_resource_keys
    if missing_private_keys:
        raise RuntimeError(
            f"MOD package omitted production encounter resources: {sorted(missing_private_keys)}"
        )
    git_path = source_dir / f"{MODULE_ROOT}.git"
    git = read_gff(git_path.read_bytes()).root
    module_ifo = read_gff((source_dir / "module.ifo").read_bytes()).root
    module_voice_id = str(module_ifo.acquire("Mod_VO_ID", "") or "")
    if module_voice_id.casefold() != str(
        TEASER_VOICE_LOOKUP["module_voice_id"]
    ).casefold():
        raise RuntimeError(
            "Exported IFO Mod_VO_ID no longer matches the teaser voice lookup."
        )
    creatures = tuple(git.acquire("Creature List", ()) or ())
    triggers = tuple(git.acquire("TriggerList", ()) or ())
    cameras = tuple(git.acquire("CameraList", ()) or ())
    placeables = tuple(git.acquire("Placeable List", ()) or ())
    sounds = tuple(git.acquire("SoundList", ()) or ())
    creature_resrefs = sorted(
        str(item.acquire("TemplateResRef", "") or "").lower() for item in creatures
    )
    trigger_resrefs = sorted(
        str(item.acquire("TemplateResRef", "") or "").lower() for item in triggers
    )
    trigger_geometry_readback: dict[str, dict[str, object]] = {}
    for trigger_row in triggers:
        resref = str(
            trigger_row.acquire("TemplateResRef", "") or ""
        ).casefold()
        position = tuple(
            float(trigger_row.acquire(label, 0.0) or 0.0)
            for label in ("XPosition", "YPosition", "ZPosition")
        )
        local_vertices = tuple(
            (
                float(point.acquire("PointX", 0.0) or 0.0),
                float(point.acquire("PointY", 0.0) or 0.0),
                float(point.acquire("PointZ", 0.0) or 0.0),
            )
            for point in tuple(trigger_row.acquire("Geometry", ()) or ())
        )
        if not local_vertices:
            raise RuntimeError(f"Exported trigger {resref!r} has no Geometry.")
        local_center = tuple(
            sum(vertex[axis] for vertex in local_vertices)
            / len(local_vertices)
            for axis in range(3)
        )
        if any(abs(value) > 0.17 for value in local_center):
            raise RuntimeError(
                f"Exported trigger {resref!r} Geometry is not local to Position."
            )
        trigger_geometry_readback[resref] = {
            "position": list(position),
            "local_vertices": [list(vertex) for vertex in local_vertices],
            "local_center": list(local_center),
            "world_center": [
                position[axis] + local_center[axis] for axis in range(3)
            ],
        }
    placeable_resrefs = sorted(
        str(item.acquire("TemplateResRef", "") or "").lower() for item in placeables
    )
    sound_resrefs = sorted(
        str(item.acquire("TemplateResRef", "") or "").lower() for item in sounds
    )
    camera_rows = {
        int(item.acquire("CameraID", 0) or 0): item
        for item in cameras
    }
    camera_readback = []
    for marker in CAMERA_MARKERS:
        camera_id = int(marker["camera_id"])
        item = camera_rows.get(camera_id)
        if item is None:
            raise RuntimeError(f"GIT readback missed camera {camera_id}.")
        orientation_value = item.acquire("Orientation", None)
        orientation = (
            tuple(float(value) for value in orientation_value)
            if orientation_value is not None
            else ()
        )
        expected_orientation = _camera_orientation(
            marker["position"],
            marker["target"],
        )
        pitch = float(item.acquire("Pitch", 0.0) or 0.0)
        if len(orientation) != 4 or any(
            not math.isclose(actual, expected, abs_tol=1.0e-5)
            for actual, expected in zip(orientation, expected_orientation)
        ):
            raise RuntimeError(
                f"Camera {camera_id} lost its Odyssey w,x,y,z orientation: "
                f"{orientation} != {expected_orientation}."
            )
        if not math.isclose(pitch, float(marker["pitch"]), abs_tol=1.0e-4):
            raise RuntimeError(
                f"Camera {camera_id} pitch changed: {pitch} != {marker['pitch']}."
            )
        camera_readback.append(
            {
                "camera_id": camera_id,
                "orientation_wxyz": list(orientation),
                "pitch_degrees": pitch,
            }
        )

    room_model = load_kotor_model_from_bytes(
        (source_dir / f"{playable_room}.mdl").read_bytes(),
        (source_dir / f"{playable_room}.mdx").read_bytes(),
        resref=playable_room,
    )
    if room_model is None:
        raise RuntimeError(f"Could not parse exported room model {playable_room}.mdl/.mdx.")
    room_nodes = []
    node_stack = [room_model.root_node]
    while node_stack:
        node = node_stack.pop()
        room_nodes.append(node)
        node_stack.extend(tuple(getattr(node, "children", ()) or ()))
    lightmapped_nodes = tuple(
        {
            "name": str(getattr(node, "name", "") or ""),
            "lightmap": str(getattr(node, "lightmap", "") or ""),
            "vertex_count": len(tuple(getattr(node, "vertices", ()) or ())),
            "lightmap_uv_count": len(tuple(getattr(node, "uvs_lm", ()) or ())),
        }
        for node in room_nodes
        if str(getattr(node, "lightmap", "") or "")
    )
    if len(lightmapped_nodes) != 8:
        raise RuntimeError(
            f"Exported room retained {len(lightmapped_nodes)} lightmapped surfaces; "
            "stock 402dxna requires 8."
        )
    if any(
        row["lightmap"] != "402dxna_lm0"
        or row["lightmap_uv_count"] != row["vertex_count"]
        for row in lightmapped_nodes
    ):
        raise RuntimeError(
            "Exported room lost a stock 402dxna lightmap name or UV2 channel."
        )
    readback = {
        "creature_resrefs": creature_resrefs,
        "trigger_resrefs": trigger_resrefs,
        "placeable_resrefs": placeable_resrefs,
        "sound_resrefs": sound_resrefs,
        "camera_count": len(cameras),
        "camera_ids": sorted(int(item.acquire("CameraID", 0) or 0) for item in cameras),
        "camera_rows": camera_readback,
        "trigger_geometry": trigger_geometry_readback,
        "module_voice_id": module_voice_id,
        "lightmapped_room_surfaces": list(lightmapped_nodes),
        "parsed_gff": list(export.package_verification.parsed_gff),
        "parsed_wok": list(export.package_verification.parsed_wok),
    }
    required_creatures = {
        PRODUCTION_XARIA_TEMPLATE,
        *PRODUCTION_WRAID_TEMPLATES,
    }
    if not required_creatures.issubset(creature_resrefs):
        raise RuntimeError(
            f"GIT readback missed creatures: {required_creatures - set(creature_resrefs)}"
        )
    if {PRODUCTION_TRIGGER_TEMPLATE, "newtransition"} - set(trigger_resrefs):
        raise RuntimeError("GIT readback missed the encounter or exit trigger.")
    if placeable_resrefs != sorted(
        (PRODUCTION_DIRECTOR_TEMPLATE, PRODUCTION_TIMELINE_TEMPLATE)
    ):
        raise RuntimeError(
            "GIT readback lost the production director or timeline owner."
        )
    if sound_resrefs != sorted(template for template, _tag in PRODUCTION_SOUND_OBJECTS):
        raise RuntimeError("GIT readback lost a production effect-sound anchor.")
    production_director = read_utp(
        (source_dir / f"{PRODUCTION_DIRECTOR_TEMPLATE}.utp").read_bytes()
    )
    if (
        str(production_director.tag) != PRODUCTION_DIRECTOR_TAG
        or str(production_director.conversation).casefold()
        != PRODUCTION_CAMERA_DIALOGUE
        or not bool(production_director.static)
        or bool(production_director.useable)
    ):
        raise RuntimeError(
            "Production director lost its tag, dialogue, or static/invisible contract."
        )
    production_timeline = read_utp(
        (source_dir / f"{PRODUCTION_TIMELINE_TEMPLATE}.utp").read_bytes()
    )
    if (
        str(production_timeline.tag) != PRODUCTION_TIMELINE_TAG
        or str(production_timeline.conversation)
        or not bool(production_timeline.static)
        or bool(production_timeline.useable)
    ):
        raise RuntimeError("Production timeline lost its inert placeable contract.")
    for template_resref, expected_tag in PRODUCTION_SOUND_OBJECTS:
        sound = read_uts((source_dir / f"{template_resref}.uts").read_bytes())
        if str(sound.tag) != expected_tag:
            raise RuntimeError(
                f"Production sound {template_resref} lost tag {expected_tag}."
            )
    if len(cameras) != len(CAMERA_MARKERS):
        raise RuntimeError(
            f"GIT readback found {len(cameras)} cameras, expected {len(CAMERA_MARKERS)}."
        )
    production_xaria = read_utc(
        (source_dir / f"{PRODUCTION_XARIA_TEMPLATE}.utc").read_bytes()
    )
    if (
        str(production_xaria.tag) != PRODUCTION_XARIA_TAG
        or int(production_xaria.appearance_id) != 725
        or not bool(production_xaria.interruptable)
        or not bool(production_xaria.plot)
        or not bool(production_xaria.min1_hp)
        or str(production_xaria.on_heartbeat).casefold() != "kxar_heart"
        or str(production_xaria.on_dialog).casefold() != "kxar_click"
        or str(production_xaria.on_end_dialog).casefold() != "kxar_enddlg"
    ):
        raise RuntimeError(
            "Production Xaria UTC lost its accepted appearance/event contract."
        )
    production_wraids = {
        resref: read_utc((source_dir / f"{resref}.utc").read_bytes())
        for resref in PRODUCTION_WRAID_TEMPLATES
    }
    if any(
        int(wraid.appearance_id) != 726
        or not bool(wraid.interruptable)
        or str(wraid.on_heartbeat).casefold() != "kxar_wheart"
        or str(wraid.on_spell).casefold() != "kxar_wspell"
        or str(wraid.on_death).casefold() != "kxar_wdeath"
        for wraid in production_wraids.values()
    ):
        raise RuntimeError(
            "Production Nydak UTCs lost their accepted appearance/event contract."
        )
    if {
        str(production_wraids[resref].tag)
        for resref in PRODUCTION_WRAID_TEMPLATES
    } != set(PRODUCTION_WRAID_TAGS):
        raise RuntimeError(
            "Production Nydak UTCs lost their accepted encounter tags."
        )
    production_trigger = read_utt(
        (source_dir / f"{PRODUCTION_TRIGGER_TEMPLATE}.utt").read_bytes()
    )
    if (
        str(production_trigger.tag) != PRODUCTION_TRIGGER_TAG
        or str(production_trigger.on_enter).casefold() != "kxar_start"
    ):
        raise RuntimeError("Production trigger does not dispatch kxar_start.")
    production_camera_dialogue = read_dlg(
        (source_dir / f"{PRODUCTION_CAMERA_DIALOGUE}.dlg").read_bytes()
    )
    dialogue_entries = list(production_camera_dialogue.all_entries(as_sorted=True))
    dialogue_replies = list(production_camera_dialogue.all_replies(as_sorted=True))
    dialogue_graph_ok = bool(
        len(production_camera_dialogue.starters) == 1
        and len(dialogue_entries) == 3
        and len(dialogue_replies) == 3
        and str(production_camera_dialogue.on_end).casefold() == "kxar_camend"
        and str(production_camera_dialogue.on_abort).casefold() == "kxar_camend"
        and production_camera_dialogue.starters[0].node is dialogue_entries[0]
        and [entry.camera_id for entry in dialogue_entries] == [111, 112, 113]
        and [entry.camera_angle for entry in dialogue_entries] == [6, 6, 6]
        and [str(entry.script1).casefold() for entry in dialogue_entries]
        == ["kxar_cam1", "", ""]
        and [entry.delay for entry in dialogue_entries] == [3, 7, 8]
    )
    if not dialogue_graph_ok:
        raise RuntimeError(
            "Production camera DLG lost its accepted 111/112/113 timeline."
        )
    readback["dialogue"] = {
        "skippable": production_camera_dialogue.skippable,
        "entry_camera_ids": [entry.camera_id for entry in dialogue_entries],
        "entry_scripts": [str(entry.script1) for entry in dialogue_entries],
        "entry_delays": [entry.delay for entry in dialogue_entries],
        "on_end_script": str(production_camera_dialogue.on_end),
        "on_abort_script": str(production_camera_dialogue.on_abort),
        "silent_combat_entry_count": sum(
            entry.text.stringref == -1 and not entry.text.get(0, 0)
            for entry in dialogue_entries
        ),
        "production_handoff_entry_count": 0,
        "independent_post_combat_handoff": True,
        "post_combat_voice_resrefs": [],
        "production_dialogue_resref": PRODUCTION_DIALOGUE,
        "voice_lookup": dict(TEASER_VOICE_LOOKUP),
        "accepted_production_spine": True,
        "graph_ok": dialogue_graph_ok,
    }
    readback["encounter_schema"] = {
        "version": "production-plcaa-20260806",
        "states": {"waiting": 0, "running": 1, "finished": 2, "recruited": 3},
        "camera_owned_first_combat_beat": True,
        "camera_dialogue_owns_combat_timeline": True,
        "trigger_retained_until_finished": True,
        "wraid_on_death_progression": True,
        "heartbeat_start_recovery": True,
        "production_dialogue_handoff": True,
        "accepted_production_spine": True,
    }
    if any(
        not (source_dir / f"{resref}.ncs").read_bytes().startswith(b"NCS V1.0")
        for resref in PRODUCTION_ENCOUNTER_NCS
    ):
        raise RuntimeError("An accepted production encounter script is not NCS.")
    private_resource_evidence = tuple(
        {
            "resref": resref,
            "restype": restype,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for resref, restype, data in private_resources
    )
    return {
        "controller": controller,
        "authored": authored,
        "combined_wok": combined.wok,
        "playable_room": playable_room,
        "dressing_rooms": tuple(dressing_rooms),
        "project_validation": project_validation,
        "placement_validation": placement_validation,
        "spatial_audit": audit_spatial_design(build_spatial_plan()),
        "lane_audit": lane_audit,
        "pie": pie,
        "export": export,
        "pack_manifest": pack_manifest,
        "semantic_rows": semantic_rows,
        "semantic_digest": _semantic_digest(semantic_rows),
        "readback": readback,
        "trigger_route_coverage": trigger_route_coverage,
        "trigger_walkmesh_audit": trigger_walkmesh_audit,
        "private_resource_evidence": private_resource_evidence,
        "external_dependency_evidence": external_dependency_evidence,
        "kmap_path": kmap_path,
        "module_path": Path(export.module_path),
    }


def _copy_candidate(first: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first["kmap_path"], KMAP_PATH)
    shutil.copy2(first["module_path"], MOD_PATH)
    source_manifest = Path(first["export"].manifest_path)
    shutil.copy2(source_manifest, ARTIFACT_DIR / f"{MODULE_ROOT}_pack_manifest.json")


def _require_deterministic_build_pair(
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    if first["semantic_digest"] != second["semantic_digest"]:
        raise RuntimeError(
            "Two clean builds produced different resource manifests: "
            f"{first['semantic_digest']} != {second['semantic_digest']}"
        )
    if (
        first["external_dependency_evidence"]
        != second["external_dependency_evidence"]
    ):
        raise RuntimeError(
            "The Xaria external dependency snapshot changed between "
            "the two clean teaser builds."
        )


def main() -> int:
    from src.core.modules.map_studio_spatial_design import spatial_design_placement_ledger

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_dependency_root = _configured_candidate_dependency_root()
    with tempfile.TemporaryDirectory(prefix="xartease_build_a_") as first_root:
        with tempfile.TemporaryDirectory(prefix="xartease_build_b_") as second_root:
            first = _build_once(
                Path(first_root),
                candidate_dependency_root=candidate_dependency_root,
            )
            second = _build_once(
                Path(second_root),
                candidate_dependency_root=candidate_dependency_root,
            )
            _require_deterministic_build_pair(first, second)
            deterministic = True
            _copy_candidate(first)
            _draw_blueprint(first["combined_wok"], BLUEPRINT_PATH)
            _draw_storyboard(STORYBOARD_PATH)
            UI_WORKFLOW_PATH.write_text(_manual_workflow(), encoding="utf-8")
            kmap_sha256 = hashlib.sha256(KMAP_PATH.read_bytes()).hexdigest()
            mod_sha256 = hashlib.sha256(MOD_PATH.read_bytes()).hexdigest()
            proof = {
                "result": "PASS",
                "candidate": "Xaria: Ichor in the Deep Grove",
                "module_root": MODULE_ROOT,
                "game": SOURCE_GAME,
                "source": {
                    "module": SOURCE_MODULE,
                    "room": SOURCE_ROOM,
                    "environment_piece": ROOM_PIECE_ID,
                    "asset_policy": "installed KOTOR II assets only",
                    "copied_textures": [],
                    "k1_asset_bytes": False,
                },
                "standalone": False,
                "dependency_contract": {
                    "bundled_production_resources": [
                        dict(row) for row in BUNDLED_ENCOUNTER_RESOURCES
                    ],
                    "bundled_production_resource_evidence": list(
                        first["private_resource_evidence"]
                    ),
                    "bundled_private_resources": [
                        dict(row) for row in BUNDLED_ENCOUNTER_RESOURCES
                    ],
                    "bundled_private_resource_evidence": list(
                        first["private_resource_evidence"]
                    ),
                    "external_runtime_dependencies": [
                        dict(row) for row in ENCOUNTER_DEPENDENCIES
                    ],
                    "external_dependency_evidence": list(
                        first["external_dependency_evidence"]
                    ),
                    "voice_lookup": dict(TEASER_VOICE_LOOKUP),
                    "production_recruitment_state_access": True,
                    "production_resource_resrefs_bundled": sorted(
                        {str(row["resref"]) for row in BUNDLED_ENCOUNTER_RESOURCES}
                    ),
                    "honesty": (
                        "xartease bundles the exact accepted PLCaa encounter spine, but it "
                        "is not standalone: Xaria models, the full recruitment dialogue, "
                        "global 2DA rows, runtime hooks, voice files, and the installed "
                        "KOTOR II library remain required."
                    ),
                },
                "separate_from_plcaa": True,
                "playable_room": first["playable_room"],
                "visual_dressing_rooms": list(first["dressing_rooms"]),
                "visual_dressing_has_collision": False,
                "combined_wok": {
                    "vertex_count": len(tuple(first["combined_wok"].verts or ())),
                    "face_count": len(tuple(first["combined_wok"].faces or ())),
                    "blocking_issues": [],
                },
                "spatial_audit": {
                    "ok": first["spatial_audit"].ok,
                    "zone_count": first["spatial_audit"].zone_count,
                    "path_count": first["spatial_audit"].path_count,
                    "placement_count": first["spatial_audit"].placement_count,
                    "warnings": list(first["spatial_audit"].warnings),
                },
                "placement_walkmesh": {
                    "ok": first["placement_validation"].ok,
                    "blocking_issues": list(first["placement_validation"].blocking_issues),
                    "warnings": list(first["placement_validation"].warnings),
                    "checks": [
                        {
                            "label": check.label,
                            "ok": check.ok,
                            "position": list(check.position),
                            "surface_id": check.surface_id,
                            "message": check.message,
                        }
                        for check in first["placement_validation"].checks
                    ],
                },
                "pie": first["pie"],
                "encounter_start_coverage": {
                    "route_metres_inside_trigger": round(
                        float(first["trigger_route_coverage"]), 6
                    ),
                    "minimum_required_metres": (
                        ENCOUNTER_TRIGGER_MIN_ROUTE_COVERAGE_METRES
                    ),
                    "proximity_fallback_metres": ENCOUNTER_PROXIMITY_METRES,
                    "heartbeat_proximity_fallback_enabled": False,
                    "click_fallback_script": "kxar_click",
                    "walkmesh_vertices": list(
                        first["trigger_walkmesh_audit"]
                    ),
                },
                "showcase_beats": [dict(row) for row in SHOWCASE_BEATS],
                "showcase_lane_audit": list(first["lane_audit"]),
                "camera_markers": [dict(row) for row in CAMERA_MARKERS],
                "readback": first["readback"],
                "export": {
                    "ok": first["export"].ok,
                    "package_readback_ok": first["export"].package_verification.ok,
                    "warnings": list(first["export"].warnings),
                    "semantic_resource_digest": first["semantic_digest"],
                    "two_clean_builds_match": deterministic,
                    "resource_count": len(first["semantic_rows"]),
                },
                "artifacts": {
                    "kmap": KMAP_PATH.name,
                    "kmap_sha256": kmap_sha256,
                    "mod": MOD_PATH.name,
                    "mod_sha256": mod_sha256,
                    "pack_manifest": f"{MODULE_ROOT}_pack_manifest.json",
                    "blueprint": BLUEPRINT_PATH.name,
                    "storyboard": STORYBOARD_PATH.name,
                    "manual_workflow": UI_WORKFLOW_PATH.name,
                },
                "visible_proof": {
                    "status": "pending retail-game capture",
                    "structural_previews": [BLUEPRINT_PATH.name, STORYBOARD_PATH.name],
                    "retail_screenshots": [],
                    "honesty": "PNG files are authored-layout diagrams, not KOTOR runtime evidence.",
                },
                "remaining_retail_gates": [
                    "Stage the full Xaria runtime dependency set transactionally with the game closed.",
                    f"Install this non-standalone candidate, then warp {MODULE_ROOT}.",
                    "Confirm entry, both route legs, and the stock-Dxun exit.",
                    "Confirm Miststep Ambush, Ichor Lightning, and Ichor Drain fire in order.",
                    "Enter once with production Xaria already recruited and prove the encounter still starts without changing recruitment.",
                    "Confirm front/rear hair physics, emissive eyes, green mist, facial animation, and close-up framing.",
                ],
            }
            PROOF_PATH.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
            manifest = {
                "schema": "ghost-studio:xaria-teaser-candidate:1",
                "module_root": MODULE_ROOT,
                "game": SOURCE_GAME,
                "standalone": False,
                "bundled_production_resources": [
                    dict(row) for row in BUNDLED_ENCOUNTER_RESOURCES
                ],
                "bundled_production_resource_evidence": list(
                    first["private_resource_evidence"]
                ),
                "bundled_private_resources": [
                    dict(row) for row in BUNDLED_ENCOUNTER_RESOURCES
                ],
                "bundled_private_resource_evidence": list(
                    first["private_resource_evidence"]
                ),
                "runtime_dependencies": [dict(row) for row in ENCOUNTER_DEPENDENCIES],
                "external_dependency_evidence": list(
                    first["external_dependency_evidence"]
                ),
                "voice_lookup": dict(TEASER_VOICE_LOOKUP),
                "production_recruitment_state_access": True,
                "production_resource_resrefs_bundled": sorted(
                    {str(row["resref"]) for row in BUNDLED_ENCOUNTER_RESOURCES}
                ),
                "world_lighting": dict(WORLD_LIGHTING),
                "entry_point": list(ENTRY_POINT),
                "exit": {
                    "position": list(EXIT_POINT),
                    "linked_to": "From_401DXN",
                    "linked_to_module": "402dxn",
                    "linked_to_flags": 2,
                    "transition_destination": 129401,
                },
                "terrain_dressing": [dict(row) for row in TERRAIN_DRESSING],
                "actor_markers": [dict(row) for row in ACTOR_MARKERS],
                "camera_markers": [dict(row) for row in CAMERA_MARKERS],
                "showcase_beats": [dict(row) for row in SHOWCASE_BEATS],
                "spatial_placement_ledger": list(
                    spatial_design_placement_ledger(build_spatial_plan())
                ),
                "semantic_resources": list(first["semantic_rows"]),
                "semantic_resource_digest": first["semantic_digest"],
                "deterministic_two_build_proof": deterministic,
                "retail_game_proof": False,
            }
            MANIFEST_PATH.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(json.dumps(proof, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
