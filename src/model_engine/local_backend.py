"""
src/model_engine/local_backend.py
==================================
Local Model Backend Wrapper.

This module wraps the existing ModelExecutor to conform to the
BaseModelBackend interface. No logic changes — pure adapter pattern.
"""

from __future__ import annotations

import logging

from src.model_engine.base_backend import BaseModelBackend
from src.model_engine.model_executor import ModelExecutor
from src.models import DocumentPage

logger = logging.getLogger(__name__)


class LocalModelBackend(BaseModelBackend):
    """
    Wrapper around ModelExecutor that implements BaseModelBackend.

    This allows the orchestrator to use the existing local inference
    engine without any code changes — it just sees a BaseModelBackend.

    Parameters
    ----------
    executor : ModelExecutor
        Pre-loaded ModelExecutor instance (load_model() already called).
    """

    def __init__(self, executor: ModelExecutor) -> None:
        self._executor = executor

    def extract_text(self, page: DocumentPage, prompt: str) -> str:
        """Delegate to ModelExecutor.extract_text()."""
        return self._executor.extract_text(page, prompt)

    def diagnose_errors(
        self, page: DocumentPage, current_text: str, prompt: str
    ) -> str:
        """Delegate to ModelExecutor.diagnose_errors()."""
        return self._executor.diagnose_errors(page, current_text, prompt)

    def filter_plan(self, raw_plan: str, prompt: str) -> str:
        """Delegate to ModelExecutor.filter_plan()."""
        return self._executor.filter_plan(raw_plan, prompt)

    def guided_refinement(
        self, page: DocumentPage, feasible_plan: str, prompt: str
    ) -> str:
        """Delegate to ModelExecutor.guided_refinement()."""
        return self._executor.guided_refinement(page, feasible_plan, prompt)

    def set_adapter(self) -> None:
        """Delegate to ModelExecutor.set_adapter()."""
        self._executor.set_adapter()

    def disable_adapter(self) -> None:
        """Delegate to ModelExecutor.disable_adapter()."""
        self._executor.disable_adapter()

    def is_adapter_available(self) -> bool:
        """
        Check if adapter is currently loaded and active.

        Returns
        -------
        bool
            True if adapter_path was provided and adapter is loaded.
        """
        return self._executor.adapter_path is not None

    def _parse_transcription(self, raw_response: str) -> str:
        """Delegate to ModelExecutor._parse_transcription()."""
        return self._executor._parse_transcription(raw_response)

    @property
    def adapter_state(self) -> str:
        """
        Return the current adapter state ("ON" or "OFF").

        This reflects the actual runtime state of the ModelExecutor.

        Returns
        -------
        str
            "ON" if adapter is currently enabled, "OFF" otherwise.
        """
        return self._executor.adapter_state
