"""Compatibility layer for the upstream G1D-infra operator controls.

This package is dependency-light and can be used by the existing
``xr_teleoperate_g1d`` JSONL/XR bridge without importing Unitree SDK modules.
"""

from .operator_events import OperatorEvent, PicoEventMapper
from .voice_announcer import VoiceAnnouncer

__all__ = ["OperatorEvent", "PicoEventMapper", "VoiceAnnouncer"]
