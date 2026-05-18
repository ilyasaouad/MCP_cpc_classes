"""shared — cross-phase utilities: base class, CPC filters."""
from .base_phase import BasePhase
from .cpc_filters import is_allocatable, filter_allocatable

__all__ = ["BasePhase", "is_allocatable", "filter_allocatable"]
