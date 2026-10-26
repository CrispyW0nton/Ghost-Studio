"""Durable, cancellable build and explicit closed-game playtest orchestration."""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import re
import threading
from uuid import uuid4

from src.io.unreal_pipeline_store import digest, file_digest, write_receipt, read_receipt, install_resources, rollback_install
from src.core.workflow.unreal_map_import import project_revision


class UnrealPipelineJobs:
    def __init__(self, project_directory, *, retail=None):
        self.root = Path(project_directory).resolve() / ".ghoststudio" / "builds"
        self.root.mkdir(parents=True, exist_ok=True)
        if retail is None:
            from src.adapters import kotor_retail_pipeline as retail
        self.retail = retail
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unreal-build")
        self._lock = threading.RLock()
        self._cancel = {}
        self._futures = {}

    def _directory(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(job_id)):
            raise ValueError("Invalid build identity.")
        return self.root / job_id

    def _update(self, identity, **values):
        with self._lock:
            path = self._directory(identity) / "job.json"
            receipt = read_receipt(path) if path.exists() else {}
            receipt.update(values, updated_at=datetime.now(timezone.utc).isoformat())
            write_receipt(path, receipt)
            return receipt

    def status(self, job_id):
        with self._lock:
            receipt = read_receipt(self._directory(job_id) / "job.json")
            if receipt["state"] in {"queued", "building", "validating"} and job_id not in self._futures:
                receipt = self._update(job_id, state="interrupted", message="Ghost closed during this build. Start a new build; previous outputs are retained.")
            installation = self._directory(job_id) / "retail" / "installation.json"
            if installation.exists():
                receipt["installation"] = read_receipt(installation)
            return receipt

    def history(self):
        return [self.status(path.parent.name) for path in sorted(self.root.glob("*/job.json"), key=lambda p:p.stat().st_mtime, reverse=True)][:30]

    def submit(self, project, resources):
        snapshot = copy.deepcopy(project)
        if not snapshot.extra_sections.get("unreal_source"):
            raise ValueError("Import an Unreal level before building this pipeline.")
        job_id = uuid4().hex
        directory = self._directory(job_id)
        directory.mkdir()
        revision = project_revision(snapshot)
        frozen_resources = tuple((str(n), str(t), bytes(b)) for n,t,b in resources)
        self._update(job_id, schema="ghoststudio.unreal-build-job/v1", job_id=job_id,
            state="queued", message="Build queued", project_revision=revision,
            source_revision=snapshot.extra_sections["unreal_source"]["source_revision"],
            cancel_requested=False, acceptance={}, game_tested=False)
        with self._lock:
            self._cancel[job_id] = threading.Event()
            self._futures[job_id] = self.worker.submit(self._build, job_id, snapshot, frozen_resources)
        return job_id

    def cancel(self, job_id):
        with self._lock:
            receipt = self.status(job_id)
            if receipt["state"] not in {"queued", "building", "validating"}:
                return receipt
            event = self._cancel.get(job_id)
            if event:
                event.set()
            return self._update(job_id, cancel_requested=True, message="Cancel requested; finishing the current compiler operation. Nothing will be installed.")

    def _checkpoint(self, job_id):
        if self._cancel[job_id].is_set():
            raise InterruptedError("Build cancelled. No game files were changed.")

    def _build(self, job_id, project, resources):
        directory = self._directory(job_id)
        try:
            self._checkpoint(job_id)
            self._update(job_id, state="building", message="Compiling geometry, walkmesh, materials and gameplay…")
            from src.core.modules.authored_module_kmap_bridge import authored_project_from_kmap_payload
            from src.core.modules.authored_module_export import AuthoredModuleExportRequest, export_authored_module_project
            from src.core.game.pykotor_gff_runtime_fix import ensure_pykotor_gff_acquire_quiet
            ensure_pykotor_gff_acquire_quiet()
            authored = authored_project_from_kmap_payload(project.extra_sections["authored_module"])
            result = export_authored_module_project(AuthoredModuleExportRequest(project=authored,
                output_dir=str(directory / "package"), extra_resources=resources, strict=True))
            self._checkpoint(job_id)
            if not result.ok:
                raise ValueError(result.message + "\n" + "\n".join(result.blocking_issues))
            self._update(job_id, state="validating", message="Verifying packaged bytes and source revision…")
            if project_revision(project) != self.status(job_id)["project_revision"]:
                raise ValueError("Project texture assets changed during the build. Build again from the current revision.")
            module = Path(result.module_path).resolve()
            files = [{"target": f"Modules/{authored.module_root}.mod", "path": str(module.relative_to(directory)), "sha256": file_digest(module)}]
            for item in tuple(getattr(result.package_result, "staged_override_resources", ()) or ()):
                path = Path(item.path).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("Compiler returned a resource outside the owned build folder.")
                files.append({"target": "Override/" + path.name, "path": str(path.relative_to(directory)), "sha256": file_digest(path)})
            self._checkpoint(job_id)
            self._update(job_id, state="ready", message="Build verified. Ready to install and playtest.",
                module=authored.module_root, files=files, warnings=list(result.warnings),
                component_count=len(authored.rooms), compiled_room_count=len(result.room_resrefs),
                module_sha256=file_digest(module), warp_command=f"warp {authored.module_root}")
        except InterruptedError as error:
            self._update(job_id, state="cancelled", message=str(error))
        except Exception as error:
            self._update(job_id, state="failed", message=str(error))

    def _files(self, job_id):
        receipt = self.status(job_id)
        directory = self._directory(job_id)
        files = []
        for row in receipt.get("files", []):
            path = (directory / row["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Build artifact escapes its job folder.")
            data = path.read_bytes()
            if digest(data) != row["sha256"]:
                raise ValueError("Built artifact changed after validation. Rebuild before installing.")
            files.append((row["target"], data))
        if not files:
            raise ValueError("This job has no verified build artifacts.")
        return files

    def install_and_play(self, job_id, profile, current_project_revision, *, launch=True):
        receipt = self.status(job_id)
        if receipt["state"] not in {"ready", "waiting_for_game_close", "installed", "launch_requested"}:
            raise ValueError("Choose a successful, verified build.")
        if receipt["project_revision"] != current_project_revision:
            raise ValueError("The scene changed after this build. Build the current scene before installing.")
        root, _ = self.retail.validate_profile(profile)
        try:
            self.retail.require_game_closed()
        except RuntimeError as error:
            return self._update(job_id, state="waiting_for_game_close", message=str(error))
        files = self._files(job_id)
        directory = self._directory(job_id)
        installation = directory / "retail" / "installation.json"
        if installation.exists():
            transaction = read_receipt(installation)
            if transaction["state"] != "installed" or Path(transaction["game_root"]).resolve() != root.resolve():
                raise ValueError("This build has a different or unfinished install transaction. Restore it or make a new build.")
            for row in transaction["files"]:
                if file_digest(root / row["target"]) != row["installed_sha256"]:
                    raise ValueError("Installed files changed outside this build. Build again or inspect recovery details.")
        else:
            self._update(job_id, state="installing", message="Backing up and installing verified module resources…")
            try:
                install_resources(root, directory / "retail", files, require_game_closed=self.retail.require_game_closed)
            except Exception as error:
                return self._update(job_id, state="recovery_required", message=str(error))
        self._update(job_id, state="installed", message="Installed bytes verified. Previous files retained for rollback.")
        if launch:
            try:
                result = self.retail.launch_playtest(profile, directory / "playtest", receipt["module"])
                return self._update(job_id, state="launch_requested", message=result["instruction"], playtest=result)
            except Exception as error:
                return self._update(job_id, state="installed", message=f"Build installed; launch needs attention: {error}")
        return self.status(job_id)

    def rollback(self, job_id):
        result = rollback_install(self._directory(job_id) / "retail/installation.json", require_game_closed=self.retail.require_game_closed)
        return self._update(job_id, state=result["state"], message="Previous files restored." if result["state"] == "rolled_back" else "Recovery needs attention; all backups retained.")

    def record_acceptance(self, job_id, checks, notes):
        receipt = self.status(job_id)
        if receipt["state"] not in {"installed", "launch_requested"}:
            raise ValueError("Install this exact build before recording its playtest.")
        expected = {"loaded", "spawn", "traversal", "save_reload"}
        if set(checks) != expected or any(type(value) is not bool for value in checks.values()):
            raise ValueError("Record each playtest observation explicitly.")
        transaction = receipt.get("installation", {})
        if transaction.get("state") != "installed" or not transaction.get("files"):
            raise ValueError("This build has no verified installed resources.")
        for row in transaction["files"]:
            if file_digest(Path(transaction["game_root"]) / row["target"]) != row["installed_sha256"]:
                raise ValueError("Installed files changed. These observations cannot qualify this build.")
        return self._update(job_id, acceptance=checks, acceptance_notes=str(notes), game_tested=all(checks.values()))
