# modqldpc/scheduling/resources.py
from __future__ import annotations
from .types import StepResourceState

def new_step_state() -> StepResourceState:
    return StepResourceState()