"""
base_phase.py — Abstract base every phase runner inherits from.

Enforces a uniform interface: __init__(cfg, kg, xml_parser, llm) + run(state).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..core.state_manager import PipelineState


class BasePhase(ABC):
    """
    Every phase runner must:
      1. Accept (cfg, kg, xml_parser, llm) in __init__.
      2. Implement run(state) -> dict  (the phase output dict).
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        kg=None,
        xml_parser=None,
        llm=None,
        **_kwargs,
    ):
        self.cfg = cfg
        self.kg = kg
        self.xml_parser = xml_parser
        self.llm = llm
        self.logger = logging.getLogger(type(self).__module__)

    @abstractmethod
    def run(self, state: PipelineState) -> Dict[str, Any]:
        """Execute phase logic. Read from state, return this phase's output dict."""
        ...
