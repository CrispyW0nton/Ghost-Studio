"""Shared secondary-motion authoring workflows."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .workflow import (
    SECONDARY_MOTION_PRESETS,
    SecondaryMotionPreview,
    compile_secondary_motion_profile,
    discover_driver_chains,
    runtime_capability_descriptor,
)

__all__ = [
    "SECONDARY_MOTION_PRESETS",
    "SecondaryMotionPreview",
    "compile_secondary_motion_profile",
    "discover_driver_chains",
    "runtime_capability_descriptor",
]
