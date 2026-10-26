"""Merge-safe KOTOR 2DA authoring for Custom Head Builder.

The Head Builder owns one modular-head row and one cloned appearance row.  It
never distributes a replacement stock table and it never changes an existing
row unless that row already contains the exact values owned by the package.
Rows are discovered by stable resource/label values on every build so another
mod can append rows without invalidating the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


HEAD_GAME_RECORD_PATCH_SCHEMA = (
    "ghostrigger.head_builder_game_record_patch.v1"
)
HEAD_GAME_RECORD_REPORT_SCHEMA = (
    "ghostrigger.head_builder_game_record_merge_report.v1"
)


class HeadGameRecordError(ValueError):
    """Raised when a live table cannot be merged without clobbering a mod."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return _sha256(payload)


def _safe_resref(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 16
        or any(not (character.isalnum() or character == "_")
               for character in text)
    ):
        raise HeadGameRecordError(
            f"{label} must be a 1..16 character KOTOR ResRef"
        )
    return text


def _read_2da(value: bytes):
    if not value:
        raise HeadGameRecordError("A required 2DA source is empty")
    try:
        from pykotor.resource.formats.twoda import read_2da

        return read_2da(value)
    except Exception as exc:
        raise HeadGameRecordError(
            f"Could not parse a required 2DA table: {exc}"
        ) from exc


def _write_2da(table: Any) -> bytes:
    try:
        from pykotor.resource.formats.twoda import bytes_2da

        result = bytes(bytes_2da(table))
        check = _read_2da(result)
        if (
            check.get_headers() != table.get_headers()
            or check.get_labels() != table.get_labels()
            or check.get_height() != table.get_height()
        ):
            raise HeadGameRecordError(
                "Merged 2DA did not preserve its headers, labels, and rows"
            )
        for row_index in range(table.get_height()):
            for header in table.get_headers():
                if (
                    check.get_cell(row_index, header)
                    != table.get_cell(row_index, header)
                ):
                    raise HeadGameRecordError(
                        "Merged 2DA cell did not survive binary readback"
                    )
        return result
    except HeadGameRecordError:
        raise
    except Exception as exc:
        raise HeadGameRecordError(
            f"Could not serialize the merged 2DA table: {exc}"
        ) from exc


def _header(table: Any, wanted: str) -> str:
    matches = [
        header
        for header in table.get_headers()
        if str(header).casefold() == str(wanted).casefold()
    ]
    if len(matches) != 1:
        raise HeadGameRecordError(
            f"The live table does not contain one '{wanted}' column"
        )
    return str(matches[0])


def _optional_header(table: Any, wanted: str) -> str:
    matches = [
        header
        for header in table.get_headers()
        if str(header).casefold() == str(wanted).casefold()
    ]
    return str(matches[0]) if len(matches) == 1 else ""


def _matches(table: Any, column: str, value: str) -> list[int]:
    wanted = str(value).casefold()
    return [
        row_index
        for row_index in range(table.get_height())
        if str(table.get_cell(row_index, column) or "").casefold()
        == wanted
    ]


def _one_match(
    table: Any,
    *,
    column: str,
    value: str,
    description: str,
) -> int:
    matches = _matches(table, column, value)
    if not matches:
        raise HeadGameRecordError(
            f"Could not re-find {description} by {column}={value!r}"
        )
    if len(matches) > 1:
        raise HeadGameRecordError(
            f"{description} is ambiguous in the live table: {matches}"
        )
    return matches[0]


def _row(table: Any, row_index: int) -> dict[str, str]:
    return {
        str(header): str(table.get_cell(row_index, header) or "")
        for header in table.get_headers()
    }


def _row_label(table: Any, row_index: int) -> str:
    return str(table.get_labels()[row_index])


def _unrelated_rows_sha256(
    table: Any,
    excluded: set[int],
) -> str:
    rows = [
        {
            "label": _row_label(table, row_index),
            "cells": _row(table, row_index),
        }
        for row_index in range(table.get_height())
        if row_index not in excluded
    ]
    return _canonical_sha256(rows)


def _require_exact_owned_row(
    table: Any,
    row_index: int,
    updates: Mapping[str, str],
    *,
    description: str,
) -> None:
    conflicts = {
        column: {
            "current": str(table.get_cell(row_index, column) or ""),
            "required": str(value),
        }
        for column, value in updates.items()
        if str(table.get_cell(row_index, column) or "").casefold()
        != str(value).casefold()
    }
    if conflicts:
        raise HeadGameRecordError(
            f"{description} already exists but is not owned by this exact "
            f"package; refusing to overwrite it: "
            + ", ".join(sorted(conflicts))
        )


@dataclass(frozen=True, slots=True)
class HeadGameRecordPatch:
    game: str
    output_head_resref: str
    texture_resref: str
    donor_head_resref: str
    body_resref: str
    appearance_donor_label: str
    appearance_label: str
    portrait_resref: str = ""
    portrait_donor_resref: str = ""
    head_texture_columns: Mapping[str, str] = field(default_factory=dict)
    portrait_texture_columns: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        game = str(self.game or "").strip().upper()
        if game not in {"K1", "K2"}:
            raise HeadGameRecordError("Head game-record patch requires K1 or K2")
        object.__setattr__(self, "game", game)
        for attribute, label in (
            ("output_head_resref", "Output head"),
            ("texture_resref", "Head texture"),
            ("donor_head_resref", "Donor head"),
            ("body_resref", "Preview body"),
        ):
            object.__setattr__(
                self,
                attribute,
                _safe_resref(getattr(self, attribute), label=label),
            )
        donor_label = str(self.appearance_donor_label or "").strip()
        custom_label = str(self.appearance_label or "").strip()
        if not donor_label or not custom_label:
            raise HeadGameRecordError(
                "Appearance donor and custom rows require stable labels"
            )
        if donor_label.casefold() == custom_label.casefold():
            raise HeadGameRecordError(
                "Custom appearance label must differ from its donor label"
            )
        object.__setattr__(
            self,
            "appearance_donor_label",
            donor_label,
        )
        object.__setattr__(self, "appearance_label", custom_label)
        portrait = str(self.portrait_resref or "").strip()
        portrait_donor = str(self.portrait_donor_resref or "").strip()
        if portrait:
            portrait = _safe_resref(
                portrait,
                label="Portrait base texture",
            )
            if not portrait_donor:
                raise HeadGameRecordError(
                    "A custom portrait requires a stable donor portrait ResRef"
                )
        if portrait_donor:
            portrait_donor = _safe_resref(
                portrait_donor,
                label="Donor portrait",
            )
        object.__setattr__(self, "portrait_resref", portrait)
        object.__setattr__(
            self,
            "portrait_donor_resref",
            portrait_donor,
        )
        cleaned = {
            str(column): _safe_resref(
                str(value),
                label=f"Head texture column {column}",
            )
            for column, value in dict(
                self.head_texture_columns or {}
            ).items()
            if str(value or "").strip()
        }
        object.__setattr__(self, "head_texture_columns", cleaned)
        portrait_columns = {
            str(column): _safe_resref(
                str(value),
                label=f"Portrait texture column {column}",
            )
            for column, value in dict(
                self.portrait_texture_columns or {}
            ).items()
            if str(value or "").strip()
        }
        if portrait_columns and not portrait:
            raise HeadGameRecordError(
                "Portrait texture columns require a custom portrait"
            )
        object.__setattr__(
            self,
            "portrait_texture_columns",
            portrait_columns,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HEAD_GAME_RECORD_PATCH_SCHEMA,
            "game": self.game,
            "output_head_resref": self.output_head_resref,
            "texture_resref": self.texture_resref,
            "donor_head_resref": self.donor_head_resref,
            "body_resref": self.body_resref,
            "appearance_donor_label": self.appearance_donor_label,
            "appearance_label": self.appearance_label,
            "portrait_resref": self.portrait_resref,
            "portrait_donor_resref": self.portrait_donor_resref,
            "head_texture_columns": dict(
                sorted(self.head_texture_columns.items())
            ),
            "portrait_texture_columns": dict(
                sorted(self.portrait_texture_columns.items())
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "HeadGameRecordPatch":
        if (
            str(payload.get("schema") or "")
            != HEAD_GAME_RECORD_PATCH_SCHEMA
        ):
            raise HeadGameRecordError(
                "Head game-record patch schema is not supported"
            )
        return cls(
            game=str(payload.get("game") or ""),
            output_head_resref=str(
                payload.get("output_head_resref") or ""
            ),
            texture_resref=str(payload.get("texture_resref") or ""),
            donor_head_resref=str(
                payload.get("donor_head_resref") or ""
            ),
            body_resref=str(payload.get("body_resref") or ""),
            appearance_donor_label=str(
                payload.get("appearance_donor_label") or ""
            ),
            appearance_label=str(
                payload.get("appearance_label") or ""
            ),
            portrait_resref=str(payload.get("portrait_resref") or ""),
            portrait_donor_resref=str(
                payload.get("portrait_donor_resref") or ""
            ),
            head_texture_columns=dict(
                payload.get("head_texture_columns") or {}
            ),
            portrait_texture_columns=dict(
                payload.get("portrait_texture_columns") or {}
            ),
        )


@dataclass(frozen=True, slots=True)
class HeadGameRecordMergeResult:
    patch: HeadGameRecordPatch
    heads_bytes: bytes = field(repr=False)
    appearance_bytes: bytes = field(repr=False)
    portraits_bytes: bytes | None = field(
        default=None,
        repr=False,
    )
    heads_row: int = -1
    appearance_row: int = -1
    portraits_row: int = -1
    report: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return (
            self.heads_row >= 0
            and self.appearance_row >= 0
            and (
                not self.patch.portrait_resref
                or self.portraits_row >= 0
            )
            and bool(self.report.get("accepted"))
        )


def merge_head_game_records(
    patch: HeadGameRecordPatch,
    *,
    heads_bytes: bytes,
    appearance_bytes: bytes,
    portraits_bytes: bytes | None = None,
) -> HeadGameRecordMergeResult:
    """Append or re-find one owned head/appearance/portrait record set."""

    if not isinstance(patch, HeadGameRecordPatch):
        raise TypeError("patch must be HeadGameRecordPatch")
    heads = _read_2da(heads_bytes)
    appearance = _read_2da(appearance_bytes)
    portraits = (
        _read_2da(portraits_bytes)
        if portraits_bytes is not None
        else None
    )

    head_column = _header(heads, "head")
    # K2 has an explicit alttexture column; retail K1 does not.  K1 resolves
    # the neutral texture from the head ResRef and its headtex* transition
    # columns, so requiring K2's column incorrectly blocks every K1 package.
    alttexture_column = _optional_header(heads, "alttexture")
    if patch.game == "K2" and not alttexture_column:
        raise HeadGameRecordError(
            "K2 heads.2da requires the alttexture column"
        )
    if (
        patch.game == "K1"
        and not alttexture_column
        and patch.texture_resref.casefold()
        != patch.output_head_resref.casefold()
    ):
        raise HeadGameRecordError(
            "K1 heads.2da without alttexture requires the head and texture "
            "ResRefs to match"
        )
    donor_head_row = _one_match(
        heads,
        column=head_column,
        value=patch.donor_head_resref,
        description="selected donor head",
    )
    donor_head_label = _row_label(heads, donor_head_row)

    appearance_label_column = _header(appearance, "label")
    normalhead_column = _header(appearance, "normalhead")
    donor_appearance_row = _one_match(
        appearance,
        column=appearance_label_column,
        value=patch.appearance_donor_label,
        description="selected donor appearance",
    )
    if (
        str(
            appearance.get_cell(
                donor_appearance_row,
                normalhead_column,
            )
            or ""
        )
        != donor_head_label
    ):
        raise HeadGameRecordError(
            "Selected appearance does not reference the selected donor "
            "head through normalhead"
        )
    body_columns = [
        str(header)
        for header in appearance.get_headers()
        if str(header).casefold() in {
            f"model{letter}" for letter in "abcdefghijklmn"
        }
    ]
    if not any(
        str(
            appearance.get_cell(donor_appearance_row, column) or ""
        ).casefold()
        == patch.body_resref.casefold()
        for column in body_columns
    ):
        raise HeadGameRecordError(
            "Selected appearance does not use the compatible preview body"
        )

    heads_before_count = heads.get_height()
    existing_heads = _matches(
        heads,
        head_column,
        patch.output_head_resref,
    )
    if len(existing_heads) > 1:
        raise HeadGameRecordError(
            "Live heads.2da contains duplicate rows for the output head"
        )
    head_updates = {head_column: patch.output_head_resref}
    if alttexture_column:
        head_updates[alttexture_column] = patch.texture_resref
    for requested_column, value in patch.head_texture_columns.items():
        actual_column = _header(heads, requested_column)
        head_updates[actual_column] = value
    if existing_heads:
        heads_row = existing_heads[0]
        _require_exact_owned_row(
            heads,
            heads_row,
            head_updates,
            description="Output head row",
        )
        heads_action = "reuse_exact"
    else:
        heads_row = heads.add_row(
            str(heads.get_height()),
            dict(head_updates),
        )
        heads_action = "append"
    head_label = _row_label(heads, heads_row)

    appearance_before_count = appearance.get_height()
    existing_appearance = _matches(
        appearance,
        appearance_label_column,
        patch.appearance_label,
    )
    if len(existing_appearance) > 1:
        raise HeadGameRecordError(
            "Live appearance.2da contains duplicate custom stable labels"
        )
    appearance_updates = {
        appearance_label_column: patch.appearance_label,
        normalhead_column: head_label,
    }
    portrait_column = _optional_header(appearance, "portrait")
    if patch.portrait_resref and portrait_column:
        appearance_updates[portrait_column] = patch.portrait_resref
    if existing_appearance:
        appearance_row = existing_appearance[0]
        _require_exact_owned_row(
            appearance,
            appearance_row,
            appearance_updates,
            description="Custom appearance row",
        )
        appearance_action = "reuse_exact"
    else:
        appearance_row = appearance.copy_row(
            appearance.get_row(donor_appearance_row),
            str(appearance.get_height()),
            dict(appearance_updates),
        )
        appearance_action = "clone_append"
    appearance_label = _row_label(appearance, appearance_row)

    portraits_row = -1
    portraits_action = "not_requested"
    portraits_before_count = (
        portraits.get_height() if portraits is not None else 0
    )
    portrait_updates: dict[str, str] = {}
    added_portrait_columns: set[str] = set()
    donor_portrait_row = -1
    if patch.portrait_resref:
        if portraits is None:
            raise HeadGameRecordError(
                "A custom portrait was requested but portraits.2da is unavailable"
            )
        base_column = _header(portraits, "baseresref")
        donor_portrait_row = _one_match(
            portraits,
            column=base_column,
            value=patch.portrait_donor_resref,
            description="selected donor portrait",
        )
        portrait_updates[base_column] = patch.portrait_resref
        for requested_column, value in patch.portrait_texture_columns.items():
            actual_column = _optional_header(portraits, requested_column)
            if not actual_column:
                # K1's retail portraits.2da predates light-side portrait
                # variants.  Runtime patches can opt into BaseResRefG/VG, so
                # custom packages must be able to add those explicit columns
                # without weakening the required stock/dark-side schema.
                if (
                    patch.game == "K1"
                    and requested_column.casefold()
                    in {"baseresrefg", "baseresrefvg"}
                ):
                    portraits.add_column(requested_column)
                    actual_column = requested_column
                    added_portrait_columns.add(actual_column)
                else:
                    actual_column = _header(portraits, requested_column)
            portrait_updates[actual_column] = value
        for requested in (
            "appearancenumber",
            "appearance_s",
            "appearance_l",
        ):
            column = _optional_header(portraits, requested)
            if column:
                portrait_updates[column] = appearance_label
        existing_portraits = _matches(
            portraits,
            base_column,
            patch.portrait_resref,
        )
        if len(existing_portraits) > 1:
            raise HeadGameRecordError(
                "Live portraits.2da contains duplicate custom portrait rows"
            )
        if existing_portraits:
            portraits_row = existing_portraits[0]
            _require_exact_owned_row(
                portraits,
                portraits_row,
                {
                    column: value
                    for column, value in portrait_updates.items()
                    if column not in added_portrait_columns
                },
                description="Custom portrait row",
            )
            for column in added_portrait_columns:
                portraits.set_cell(
                    portraits_row,
                    column,
                    portrait_updates[column],
                )
            portraits_action = (
                "upgrade_additive_columns"
                if added_portrait_columns
                else "reuse_exact"
            )
        else:
            portraits_row = portraits.copy_row(
                portraits.get_row(donor_portrait_row),
                str(portraits.get_height()),
                dict(portrait_updates),
            )
            portraits_action = "clone_append"

    merged_heads = _write_2da(heads)
    merged_appearance = _write_2da(appearance)
    merged_portraits = (
        _write_2da(portraits)
        if portraits is not None and patch.portrait_resref
        else portraits_bytes
    )
    heads_check = _read_2da(merged_heads)
    appearance_check = _read_2da(merged_appearance)
    if (
        str(heads_check.get_cell(heads_row, head_column) or "").casefold()
        != patch.output_head_resref.casefold()
        or str(
            appearance_check.get_cell(
                appearance_row,
                normalhead_column,
            )
            or ""
        )
        != _row_label(heads_check, heads_row)
    ):
        raise HeadGameRecordError(
            "Merged head/appearance link did not survive 2DA readback"
        )

    report: dict[str, Any] = {
        "schema": HEAD_GAME_RECORD_REPORT_SCHEMA,
        "accepted": True,
        "patch": patch.to_dict(),
        "rows": {
            "heads": {
                "action": heads_action,
                "row_index": heads_row,
                "row_label": head_label,
                "before_count": heads_before_count,
                "after_count": heads.get_height(),
                "after": _row(heads, heads_row),
            },
            "appearance": {
                "action": appearance_action,
                "row_index": appearance_row,
                "row_label": appearance_label,
                "donor_row_label": _row_label(
                    appearance,
                    donor_appearance_row,
                ),
                "before_count": appearance_before_count,
                "after_count": appearance.get_height(),
                "after": _row(appearance, appearance_row),
            },
            "portraits": {
                "action": portraits_action,
                "row_index": portraits_row,
                "row_label": (
                    _row_label(portraits, portraits_row)
                    if portraits is not None and portraits_row >= 0
                    else ""
                ),
                "donor_row_label": (
                    _row_label(portraits, donor_portrait_row)
                    if portraits is not None and donor_portrait_row >= 0
                    else ""
                ),
                "before_count": portraits_before_count,
                "after_count": (
                    portraits.get_height()
                    if portraits is not None
                    else 0
                ),
                "after": (
                    _row(portraits, portraits_row)
                    if portraits is not None and portraits_row >= 0
                    else {}
                ),
            },
        },
        "donor": {
            "heads_row": donor_head_row,
            "heads_row_label": donor_head_label,
            "appearance_row": donor_appearance_row,
            "appearance_row_label": _row_label(
                appearance,
                donor_appearance_row,
            ),
        },
        "hashes": {
            "heads_before_sha256": _sha256(heads_bytes),
            "heads_after_sha256": _sha256(merged_heads),
            "appearance_before_sha256": _sha256(appearance_bytes),
            "appearance_after_sha256": _sha256(merged_appearance),
            "portraits_before_sha256": (
                _sha256(portraits_bytes)
                if portraits_bytes is not None
                else ""
            ),
            "portraits_after_sha256": (
                _sha256(merged_portraits)
                if merged_portraits is not None
                else ""
            ),
            "heads_unrelated_before_sha256": _unrelated_rows_sha256(
                _read_2da(heads_bytes),
                set(),
            ),
            "heads_unrelated_after_sha256": _unrelated_rows_sha256(
                heads,
                {heads_row} if heads_action == "append" else set(),
            ),
            "appearance_unrelated_before_sha256": (
                _unrelated_rows_sha256(
                    _read_2da(appearance_bytes),
                    set(),
                )
            ),
            "appearance_unrelated_after_sha256": (
                _unrelated_rows_sha256(
                    appearance,
                    (
                        {appearance_row}
                        if appearance_action == "clone_append"
                        else set()
                    ),
                )
            ),
        },
        "no_existing_row_modified": True,
        "stable_refind": {
            "heads": {
                "column": head_column,
                "value": patch.output_head_resref,
            },
            "appearance": {
                "column": appearance_label_column,
                "value": patch.appearance_label,
            },
            "portraits": {
                "column": "baseresref",
                "value": patch.portrait_resref,
            },
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return HeadGameRecordMergeResult(
        patch=patch,
        heads_bytes=merged_heads,
        appearance_bytes=merged_appearance,
        portraits_bytes=merged_portraits,
        heads_row=heads_row,
        appearance_row=appearance_row,
        portraits_row=portraits_row,
        report=report,
    )


def load_live_twoda(
    game_directory: str | Path,
    table_name: str,
) -> bytes:
    """Load the effective loose table or the installed KEY/BIF resource."""

    root = Path(game_directory).expanduser().resolve()
    name = Path(str(table_name or "")).stem.casefold()
    if (
        not name
        or any(not (character.isalnum() or character == "_")
               for character in name)
    ):
        raise HeadGameRecordError("Unsafe KOTOR 2DA table name")
    override = root / "Override" / f"{name}.2da"
    if override.is_file():
        return override.read_bytes()
    try:
        from pykotor.extract.installation import Installation
        from pykotor.resource.type import ResourceType

        resource = Installation(root).resource(name, ResourceType.TwoDA)
    except Exception as exc:
        raise HeadGameRecordError(
            f"Could not read {name}.2da from the selected game: {exc}"
        ) from exc
    if resource is None:
        raise HeadGameRecordError(
            f"The selected game does not contain {name}.2da"
        )
    return bytes(resource.data)


__all__ = [
    "HEAD_GAME_RECORD_PATCH_SCHEMA",
    "HEAD_GAME_RECORD_REPORT_SCHEMA",
    "HeadGameRecordError",
    "HeadGameRecordMergeResult",
    "HeadGameRecordPatch",
    "load_live_twoda",
    "merge_head_game_records",
]
