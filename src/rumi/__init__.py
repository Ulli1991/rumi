"""Speed-first N-body building blocks; no automatic precision or device configuration."""

from .state import State, validate_state

__all__ = ["State", "validate_state"]
