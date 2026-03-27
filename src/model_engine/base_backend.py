"""
src/model_engine/base_backend.py
=================================
Abstract Backend Interface for Model Execution.

This module defines the contract that all model backends must implement.
The AgenticOrchestrator depends only on this interface, not on concrete
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import DocumentPage


class BaseModelBackend(ABC):
    """
    Abstract base class defining the interface for model backends.

    The orchestrator calls these methods without knowing whether the
    backend is running local inference or using a remote API.
    """

    @abstractmethod
    def extract_text(self, page: DocumentPage, prompt: str) -> str:
        """
        Single-pass OCR extraction.

        Adapter ON equivalent if adapter exists.

        Parameters
        ----------
        page : DocumentPage
            The document page to process.
        prompt : str
            The OCR prompt.

        Returns
        -------
        str
            Extracted transcription text.
        """
        pass

    @abstractmethod
    def diagnose_errors(
        self, page: DocumentPage, current_text: str, prompt: str
    ) -> str:
        """
        Identify transcription errors and produce a correction plan.

        Adapter OFF equivalent.

        Parameters
        ----------
        page : DocumentPage
            The document page (not used in text-only mode).
        current_text : str
            Current transcription text.
        prompt : str
            The reflection prompt.

        Returns
        -------
        str
            Error diagnosis and correction plan.
        """
        pass

    @abstractmethod
    def filter_plan(self, raw_plan: str, prompt: str) -> str:
        """
        Filter infeasible corrections from a raw plan.

        Adapter OFF equivalent.

        Parameters
        ----------
        raw_plan : str
            Raw correction plan from diagnose_errors.
        prompt : str
            The filtering prompt.

        Returns
        -------
        str
            Feasible subset of the correction plan.
        """
        pass

    @abstractmethod
    def guided_refinement(
        self, page: DocumentPage, feasible_plan: str, prompt: str
    ) -> str:
        """
        Refine transcription using a feasible correction plan.

        Adapter ON equivalent.

        Parameters
        ----------
        page : DocumentPage
            The document page to process.
        feasible_plan : str
            Filtered correction plan.
        prompt : str
            The refinement prompt.

        Returns
        -------
        str
            Refined transcription text.
        """
        pass

    @abstractmethod
    def set_adapter(self) -> None:
        """
        Enable domain adapter if available.

        No-op if not applicable (e.g., API backends).
        """
        pass

    @abstractmethod
    def disable_adapter(self) -> None:
        """
        Disable domain adapter if available.

        No-op if not applicable (e.g., API backends).
        """
        pass

    @abstractmethod
    def is_adapter_available(self) -> bool:
        """
        Return True if a LoRA adapter or equivalent is loaded and usable.

        Returns
        -------
        bool
            True if adapter is available, False otherwise.
        """
        pass

    @abstractmethod
    def _parse_transcription(self, raw_response: str) -> str:
        """
        Extract the clean transcription from a raw model response.

        Handles common response patterns such as thinking tags,
        markdown separators, or introductory prose.

        Parameters
        ----------
        raw_response : str
            The full raw output from the model.

        Returns
        -------
        str
            The extracted transcription text, stripped of all preamble.
        """
        pass

    @property
    def adapter_state(self) -> str:
        """
        Return the current adapter state as "ON" or "OFF".

        This is a convenience property for logging and tracing.
        Subclasses can override if they need to track dynamic state.

        Returns
        -------
        str
            "ON" if adapter is available, "OFF" otherwise.
        """
        return "ON" if self.is_adapter_available() else "OFF"
