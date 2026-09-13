"""Shared staged export transaction helper for GhostRigger."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from src.core.project.resource_address import ResourceAddress
from src.core.validation.validation_bus import (
    ValidationBus,
    ValidationIssue,
    ValidationReport,
    ValidationSeverity,
    ValidationSubsystem,
    merge_validation_reports,
)


class ExportJobStatus(str, Enum):
    PENDING = "pending"
    PREFLIGHT_FAILED = "preflight_failed"
    WRITING = "writing"
    VERIFYING = "verifying"
    PROMOTING = "promoting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ExportOutputSpec:
    final_path: Path
    artifact_kind: str
    address: ResourceAddress | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_path", Path(self.final_path))
        object.__setattr__(self, "artifact_kind", str(self.artifact_kind or "output"))


@dataclass
class ExportJobRequest:
    job_id: str
    kind: str
    outputs: list[ExportOutputSpec]
    overwrite: bool = False
    staging_root: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    preflight_report: ValidationReport | None = None
    validation_bus_source: str | None = None

    def __post_init__(self) -> None:
        self.job_id = str(self.job_id or "")
        self.kind = str(self.kind or "")
        self.outputs = list(self.outputs or [])
        self.staging_root = Path(self.staging_root) if self.staging_root is not None else None
        self.metadata = dict(self.metadata or {})


@dataclass
class ExportJobContext:
    request: ExportJobRequest
    staging_dir: Path
    output_map: dict[Path, Path]

    def staged_path_for(self, final_path: Path) -> Path:
        wanted = _normalized_path_key(Path(final_path))
        for final, staged in self.output_map.items():
            if _normalized_path_key(final) == wanted:
                return staged
        raise KeyError(f"No staged output path registered for {final_path}")

    def write_bytes(self, path: str | Path, data: bytes) -> None:
        staged = self.staged_path_for(Path(path))
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(bytes(data or b""))

    def write_text(self, path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
        staged = self.staged_path_for(Path(path))
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(str(text), encoding=encoding)


@dataclass
class ExportJobResult:
    job_id: str
    kind: str
    status: ExportJobStatus
    outputs: list[ExportOutputSpec]
    staged_paths: dict[str, str]
    final_paths: list[Path]
    validation_report: ValidationReport
    manifest_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == ExportJobStatus.SUCCEEDED


def run_export_job(
    request: ExportJobRequest,
    *,
    writer: Callable[[ExportJobContext], None],
    verifier: Callable[[ExportJobContext], ValidationReport | None] | None = None,
    manifest_writer: Callable[[ExportJobContext, ExportJobResult], Path | None] | None = None,
    validation_bus: ValidationBus | None = None,
) -> ExportJobResult:
    """Run a staged multi-file export and promote only verified artifacts."""

    preflight = _preflight_export_request(request)
    if preflight.has_blocking:
        result = _result(
            request,
            ExportJobStatus.PREFLIGHT_FAILED,
            preflight,
            staged_paths={},
            final_paths=[],
        )
        _publish(validation_bus, request, result.validation_report)
        return result

    final_parents = _output_parents(request.outputs)
    assert final_parents
    for parent in final_parents:
        parent.mkdir(parents=True, exist_ok=True)

    staging_parent = Path(request.staging_root) if request.staging_root is not None else _default_staging_parent(request.outputs)
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".grx_{_bounded_safe_component(request.job_id, max_length=24)}_",
            dir=str(staging_parent),
        )
    )
    output_map = _staged_output_map(request.outputs, staging_dir)
    context = ExportJobContext(request=request, staging_dir=staging_dir, output_map=output_map)
    staged_paths = {str(final): str(staged) for final, staged in output_map.items()}
    backups: dict[Path, Path] = {}

    try:
        try:
            writer(context)
        except Exception as exc:
            report = _single_issue_report(
                request,
                code="export.writer.exception",
                message=f"Writer failed for export job '{request.job_id}': {exc}",
                severity=ValidationSeverity.BLOCKING,
                details={"exception_type": type(exc).__name__},
            )
            result = _result(request, ExportJobStatus.FAILED, report, staged_paths=staged_paths, final_paths=[])
            _publish(validation_bus, request, report)
            return result

        interim = _result(
            request,
            ExportJobStatus.WRITING,
            ValidationReport(source=request.validation_bus_source),
            staged_paths=staged_paths,
            final_paths=[],
        )
        manifest_path: Path | None = None
        if manifest_writer is not None:
            try:
                manifest_path = manifest_writer(context, interim)
            except Exception as exc:
                report = _single_issue_report(
                    request,
                    code="export.writer.exception",
                    message=f"Manifest writer failed for export job '{request.job_id}': {exc}",
                    severity=ValidationSeverity.BLOCKING,
                    details={"exception_type": type(exc).__name__},
                )
                result = _result(request, ExportJobStatus.FAILED, report, staged_paths=staged_paths, final_paths=[])
                _publish(validation_bus, request, report)
                return result

        missing_report = _validate_staged_outputs(context)
        if missing_report.has_blocking:
            result = _result(request, ExportJobStatus.FAILED, missing_report, staged_paths=staged_paths, final_paths=[])
            _publish(validation_bus, request, missing_report)
            return result

        verification_report = ValidationReport(source=request.validation_bus_source)
        if verifier is not None:
            try:
                verification_report = verifier(context) or ValidationReport(source=request.validation_bus_source)
            except Exception as exc:
                verification_report = _single_issue_report(
                    request,
                    code="export.verification.failed",
                    message=f"Verifier failed for export job '{request.job_id}': {exc}",
                    severity=ValidationSeverity.BLOCKING,
                    details={"exception_type": type(exc).__name__},
                )
        if verification_report.has_blocking:
            combined = merge_validation_reports(preflight, verification_report)
            result = _result(request, ExportJobStatus.FAILED, combined, staged_paths=staged_paths, final_paths=[])
            _publish(validation_bus, request, combined)
            return result

        final_paths: list[Path] = []
        try:
            for spec in request.outputs:
                final = spec.final_path
                staged = context.staged_path_for(final)
                if final.exists() and request.overwrite:
                    backup = staging_dir / f"{final.name}.ghostrigger_backup"
                    os.replace(final, backup)
                    backups[final] = backup
                if request.overwrite:
                    os.replace(staged, final)
                else:
                    # Atomic creation: a file appearing after preflight must
                    # never be replaced. Staging cleanup removes the other link.
                    os.link(staged, final)
                final_paths.append(final)
        except Exception as exc:
            rollback_messages = _rollback_promoted_outputs(final_paths, backups)
            message = f"Promotion failed for export job '{request.job_id}': {exc}"
            if rollback_messages:
                message += " Rollback issues: " + "; ".join(rollback_messages)
            report = _single_issue_report(
                request,
                code="export.promotion.failed",
                message=message,
                severity=ValidationSeverity.BLOCKING,
                details={"exception_type": type(exc).__name__, "rollback_issues": rollback_messages},
            )
            result = _result(request, ExportJobStatus.FAILED, report, staged_paths=staged_paths, final_paths=final_paths)
            _publish(validation_bus, request, report)
            return result

        success_report = merge_validation_reports(
            preflight,
            verification_report,
            _single_issue_report(
                request,
                code="export.succeeded",
                message=f"Export job '{request.job_id}' completed successfully.",
                severity=ValidationSeverity.INFO,
            ),
        )
        result = _result(
            request,
            ExportJobStatus.SUCCEEDED,
            success_report,
            staged_paths=staged_paths,
            final_paths=final_paths,
            manifest_path=_final_manifest_path(request, manifest_path),
        )
        _publish(validation_bus, request, success_report)
        return result
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _preflight_export_request(request: ExportJobRequest) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not request.job_id.strip():
        issues.append(_issue("export.preflight.blocking", "Export job ID is required."))
    if not request.kind.strip():
        issues.append(_issue("export.preflight.blocking", "Export job kind is required."))
    if not request.outputs:
        issues.append(_issue("export.preflight.blocking", "Export job requires at least one output."))

    try:
        _ensure_json_serializable(request.metadata, "Export job metadata")
    except ValueError as exc:
        issues.append(_issue("export.preflight.blocking", str(exc)))

    keys: set[str] = set()
    for spec in request.outputs:
        final = Path(spec.final_path)
        key = _normalized_path_key(final)
        if key in keys:
            issues.append(
                _issue(
                    "export.output.duplicate",
                    f"Duplicate export output path: {final}",
                    details={"path": str(final)},
                )
            )
        keys.add(key)
        if not str(spec.artifact_kind or "").strip():
            issues.append(_issue("export.preflight.blocking", f"Output {final} has no artifact kind."))
        parent = final.parent
        if parent.exists() and not parent.is_dir():
            issues.append(_issue("export.preflight.blocking", f"Output parent is not a directory: {parent}"))
        if final.exists() and not request.overwrite:
            issues.append(
                _issue(
                    "export.output.exists",
                    f"Export would overwrite existing file: {final}",
                    details={"path": str(final)},
                )
            )

    if request.preflight_report is not None:
        issues.extend(request.preflight_report.issues)

    return ValidationReport(issues=issues, source=request.validation_bus_source)


def _validate_staged_outputs(context: ExportJobContext) -> ValidationReport:
    issues: list[ValidationIssue] = []
    for spec in context.request.outputs:
        staged = context.staged_path_for(spec.final_path)
        if spec.required and not staged.exists():
            issues.append(
                _issue(
                    "export.verification.failed",
                    f"Required staged export artifact was not written: {staged}",
                    details={"final_path": str(spec.final_path), "staged_path": str(staged)},
                )
            )
    return ValidationReport(issues=issues, source=context.request.validation_bus_source)


def _result(
    request: ExportJobRequest,
    status: ExportJobStatus,
    report: ValidationReport,
    *,
    staged_paths: dict[str, str],
    final_paths: list[Path],
    manifest_path: Path | None = None,
) -> ExportJobResult:
    warnings = [
        issue.message
        for issue in report.issues
        if issue.severity == ValidationSeverity.WARNING
    ]
    return ExportJobResult(
        job_id=request.job_id,
        kind=request.kind,
        status=status,
        outputs=list(request.outputs),
        staged_paths=dict(staged_paths),
        final_paths=list(final_paths),
        validation_report=report,
        manifest_path=manifest_path,
        warnings=warnings,
        metadata=dict(request.metadata),
    )


def _single_issue_report(
    request: ExportJobRequest,
    *,
    code: str,
    message: str,
    severity: ValidationSeverity = ValidationSeverity.BLOCKING,
    details: dict[str, Any] | None = None,
) -> ValidationReport:
    return ValidationReport(
        issues=[
            ValidationIssue(
                severity=severity,
                subsystem=ValidationSubsystem.EXPORT,
                code=code,
                message=message,
                details=dict(details or {}),
                source=request.validation_bus_source,
            )
        ],
        source=request.validation_bus_source,
    )


def _issue(
    code: str,
    message: str,
    *,
    severity: ValidationSeverity = ValidationSeverity.BLOCKING,
    details: dict[str, Any] | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        subsystem=ValidationSubsystem.EXPORT,
        code=code,
        message=message,
        details=dict(details or {}),
    )


def _publish(validation_bus: ValidationBus | None, request: ExportJobRequest, report: ValidationReport) -> None:
    if validation_bus is None or not request.validation_bus_source:
        return
    if report.source != request.validation_bus_source:
        report = ValidationReport(
            issues=report.issues,
            source=request.validation_bus_source,
            metadata=dict(report.metadata),
        )
    validation_bus.publish(report, replace_source=True)


def _output_parents(outputs: list[ExportOutputSpec]) -> list[Path]:
    parents: dict[str, Path] = {}
    for spec in outputs:
        parent = spec.final_path.parent
        parents[_normalized_path_key(parent)] = parent
    return list(parents.values())


def _default_staging_parent(outputs: list[ExportOutputSpec]) -> Path:
    parents = _output_parents(outputs)
    if not parents:
        return Path(".")
    shared = _shared_output_parent(outputs)
    if shared is not None:
        return shared
    try:
        common = Path(os.path.commonpath([str(parent) for parent in parents]))
    except ValueError:
        return parents[0]
    # Never choose a drive/filesystem root merely because outputs span
    # unrelated trees; use the first output parent in that uncommon case.
    return common if common.parent != common else parents[0]


def _shared_output_parent(outputs: list[ExportOutputSpec]) -> Path | None:
    if not outputs:
        return None
    parents = {_normalized_path_key(spec.final_path.parent) for spec in outputs}
    if len(parents) != 1:
        return None
    return outputs[0].final_path.parent


def _staged_output_map(outputs: list[ExportOutputSpec], staging_dir: Path) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    used: set[str] = set()
    shared_parent = _shared_output_parent(outputs)
    for index, spec in enumerate(outputs):
        if shared_parent is not None:
            candidate = staging_dir / spec.final_path.name
        else:
            parent_slug = _staged_parent_slug(spec.final_path.parent)
            candidate = staging_dir / f"{index:04d}_{parent_slug}" / spec.final_path.name
        key = _normalized_path_key(candidate)
        if key in used:
            candidate = candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}")
            key = _normalized_path_key(candidate)
        used.add(key)
        mapping[spec.final_path] = candidate
    return mapping


def _rollback_promoted_outputs(final_paths: list[Path], backups: dict[Path, Path]) -> list[str]:
    messages: list[str] = []
    for final in reversed(final_paths):
        try:
            if final.exists():
                final.unlink()
        except OSError as exc:
            messages.append(f"could not remove promoted output {final}: {exc}")
    for final, backup in backups.items():
        try:
            if backup.exists():
                os.replace(backup, final)
        except OSError as exc:
            messages.append(f"could not restore backup {final}: {exc}")
    return messages


def _final_manifest_path(request: ExportJobRequest, staged_manifest: Path | None) -> Path | None:
    for spec in request.outputs:
        if spec.artifact_kind.lower() == "manifest":
            return spec.final_path
    return staged_manifest


def _normalized_path_key(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def _safe_job_id(job_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(job_id or "job"))
    return safe.strip("_") or "job"


def _bounded_safe_component(value: str, *, max_length: int) -> str:
    """Return a readable deterministic component without growing temp paths."""

    safe = _safe_job_id(value)
    limit = max(12, int(max_length))
    if len(safe) <= limit:
        return safe
    digest = hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{safe[: limit - 9]}_{digest}"


def _staged_parent_slug(parent: Path) -> str:
    normalized = _normalized_path_key(parent)
    leaf = Path(parent).name or "output"
    digest = hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{_bounded_safe_component(leaf, max_length=16)}_{digest}"


def _ensure_json_serializable(value: Any, context: str) -> None:
    try:
        json.dumps(value)
    except TypeError as exc:
        raise ValueError(f"{context} contains non-JSON-serializable data: {exc}") from exc
