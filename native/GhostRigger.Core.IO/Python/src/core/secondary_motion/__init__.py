"""Secondary-motion profile serialization and model-format verification."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .profile_io import (
    DanglyPayloadVerification,
    deserialize_secondary_motion_profile,
    read_secondary_motion_profile,
    serialize_secondary_motion_profile,
    verify_raw_dangly_payload,
    write_secondary_motion_profile,
)

__all__ = [
    "DanglyPayloadVerification",
    "deserialize_secondary_motion_profile",
    "read_secondary_motion_profile",
    "serialize_secondary_motion_profile",
    "verify_raw_dangly_payload",
    "write_secondary_motion_profile",
]
