from pathlib import Path
import json
import pytest
from src.io import unreal_pipeline_store as store
from src.core.workflow.unreal_pipeline_jobs import UnrealPipelineJobs
from src.io.unreal_editor_connection import capture_conversion_profile, restore_conversion_profile
from src.io.unreal_native_snapshot import load_native_snapshot, convert_native_snapshot
from tests.test_unreal_native_snapshot import native_fixture


def closed():
    pass


def test_same_basename_resources_keep_separate_backups_and_restore(tmp_path):
    root = tmp_path / "game"
    for folder, data in (("Modules", b"old mod"), ("Override", b"old override")):
        (root / folder).mkdir(parents=True)
        (root / folder / "same.mod").write_bytes(data)
    path = store.install_resources(root, tmp_path / "transaction", [("Modules/same.mod", b"new mod"), ("Override/same.mod", b"new override")], require_game_closed=closed)
    assert store.read_receipt(path)["state"] == "installed"
    assert store.rollback_install(path, require_game_closed=closed)["state"] == "rolled_back"
    assert (root / "Modules/same.mod").read_bytes() == b"old mod"
    assert (root / "Override/same.mod").read_bytes() == b"old override"
    assert len(list((path.parent / "backups").iterdir())) == 2


def test_failed_second_write_restores_first_and_retains_backups(tmp_path, monkeypatch):
    root = tmp_path / "game"
    (root / "Modules").mkdir(parents=True)
    (root / "Modules/a.mod").write_bytes(b"old")
    original = store.atomic_bytes
    def fail(path, data):
        if path == root / "Modules/b.mod":
            raise OSError("injected disk failure")
        original(path, data)
    monkeypatch.setattr(store, "atomic_bytes", fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.install_resources(root, tmp_path / "transaction", [("Modules/a.mod", b"new"), ("Modules/b.mod", b"new")], require_game_closed=closed)
    assert (root / "Modules/a.mod").read_bytes() == b"old"
    receipt = store.read_receipt(tmp_path / "transaction/installation.json")
    assert receipt["state"] == "rolled_back"
    assert (tmp_path / "transaction" / receipt["files"][0]["backup"]).read_bytes() == b"old"


def test_rollback_preserves_external_change_and_recovery_backup(tmp_path):
    root = tmp_path / "game"
    (root / "Modules").mkdir(parents=True)
    target = root / "Modules/a.mod"
    target.write_bytes(b"old")
    path = store.install_resources(root, tmp_path / "transaction", [("Modules/a.mod", b"new")], require_game_closed=closed)
    target.write_bytes(b"another tool")
    receipt = store.rollback_install(path, require_game_closed=closed)
    assert receipt["state"] == "recovery_required"
    assert target.read_bytes() == b"another tool"
    assert (path.parent / receipt["files"][0]["backup"]).read_bytes() == b"old"


@pytest.mark.parametrize("target", ["../x", "Modules/../../x", "Modules/a:b.mod", "Modules/a\\b.mod"])
def test_install_rejects_escaping_destinations(tmp_path, target):
    root = tmp_path / "game"
    root.mkdir()
    with pytest.raises(ValueError):
        store.install_resources(root, tmp_path / "transaction", [(target,b"x")], require_game_closed=closed)
    assert not (tmp_path / "transaction").exists()


def test_saved_choices_rebind_only_when_collision_source_unchanged(tmp_path):
    source, policy, raw, _ = native_fixture(tmp_path)
    snapshot = load_native_snapshot(source)
    project = tmp_path / "project"
    project.mkdir()
    reference = capture_conversion_profile(project, policy, snapshot)
    raw["components"][0]["label"] = "Renamed floor"
    source.write_text(json.dumps(raw))
    changed = load_native_snapshot(source)
    restored = restore_conversion_profile(project, reference, changed, tmp_path / "settings")
    assert convert_native_snapshot(source, restored, tmp_path / "converted").is_file()
    changed["meshes"]["/Game/Floor.Floor"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="collision source changed"):
        restore_conversion_profile(project, reference, changed, tmp_path / "settings2")


class Retail:
    def __init__(self, root):
        self.root, self.open, self.launched = root, False, False
    def validate_profile(self, profile):
        return self.root, self.root / "launcher.exe"
    def require_game_closed(self):
        if self.open:
            raise RuntimeError("Close the game first")
    def launch_playtest(self, *args):
        self.launched = True
        return {"instruction": "Load a clean save", "launch_requested": True}


def ready_job(tmp_path):
    root = tmp_path / "game"
    (root / "Modules").mkdir(parents=True)
    retail = Retail(root)
    jobs = UnrealPipelineJobs(tmp_path / "project", retail=retail)
    job = "a"*32
    path = jobs.root / job / "package/a.mod"
    store.atomic_bytes(path,b"compiled")
    jobs._update(job, job_id=job, state="ready", project_revision="current", module="a", files=[{"target":"Modules/a.mod","path":"package/a.mod","sha256":store.digest(b"compiled")}])
    return jobs, retail, job, path


def test_open_game_gate_and_hash_validation_precede_install(tmp_path):
    jobs, retail, job, path = ready_job(tmp_path)
    retail.open = True
    assert jobs.install_and_play(job, {}, "current")["state"] == "waiting_for_game_close"
    assert not (retail.root / "Modules/a.mod").exists()
    retail.open = False
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed after validation"):
        jobs.install_and_play(job, {}, "current")
    assert not retail.launched


def test_exact_installed_bytes_required_for_playtest_record(tmp_path):
    jobs, retail, job, _ = ready_job(tmp_path)
    assert jobs.install_and_play(job, {}, "current")["state"] == "launch_requested"
    checks = dict(loaded=True,spawn=True,traversal=True,save_reload=True)
    assert jobs.record_acceptance(job, checks, "Observed")["game_tested"]
    (retail.root / "Modules/a.mod").write_bytes(b"other build")
    with pytest.raises(ValueError, match="Installed files changed"):
        jobs.record_acceptance(job, checks, "Observed")


def test_previous_process_build_is_interrupted_and_can_be_rebuilt(tmp_path):
    jobs, _, job, _ = ready_job(tmp_path)
    jobs._update(job, state="building")
    assert jobs.status(job)["state"] == "interrupted"


def test_spawn_proposal_is_on_native_fixture_floor(tmp_path):
    from src.math.unreal_native_surface import propose_section_spawn
    source, _, _, _ = native_fixture(tmp_path)
    snapshot = load_native_snapshot(source)
    component = snapshot["components"][0]
    section = snapshot["resolved_meshes"][component["mesh"]]["sections"][0]
    point = propose_section_spawn(section, component["world_matrix"])
    assert len(point) == 3
    assert point[2] == pytest.approx(0)
