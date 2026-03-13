"""
src/models.py
=============
Central data-model definitions for the Agentic OCR Framework.

All inter-module hand-offs use these typed entities, ensuring strict
contract enforcement between pipeline stages.  Built with pydantic v2
for runtime validation and easy JSON serialisation.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ExecutionMode(str, enum.Enum):
    """
    The four supported pipeline execution modes described in the System
    Design Document §2.3.

    Attributes
    ----------
    BASE_ONE_SHOT:
        Single-pass extraction using the base VLM, no adapters.
    BASE_REACT:
        Multi-turn agentic loop using the base VLM, no adapters.
    ADAPTER_ONE_SHOT:
        Single-pass extraction with domain-specific LoRA adapters loaded.
    ADAPTER_REACT:
        Full pipeline — LoRA adapters + iterative Capability & Memory
        Reflection loop (the primary research mode).
    """
    BASE_ONE_SHOT    = "Base_OneShot"
    BASE_REACT       = "Base_ReAct"
    ADAPTER_ONE_SHOT = "Adapter_OneShot"
    ADAPTER_REACT    = "Adapter_ReAct"


# ---------------------------------------------------------------------------
# Pipeline entities
# ---------------------------------------------------------------------------

class DocumentPage(BaseModel):
    """
    Represents a single page extracted from a source PDF.

    Produced by :class:`~src.ingestion.pdf_handler.PDFHandler` and
    consumed by :class:`~src.model_engine.model_executor.ModelExecutor`.

    Attributes
    ----------
    page_id : str
        Unique identifier, typically ``"<doc_stem>_page_<N>"``.
    image_path : str
        Absolute path to the rasterised PNG/JPEG on disk.  The raw
        binary is *not* stored here to keep objects serialisable.
    metadata : dict[str, Any]
        Arbitrary document-level metadata (source year, archive ref,
        original filename, page number, etc.).
    """
    page_id    : str
    image_path : str
    metadata   : dict[str, Any] = Field(default_factory=dict)


class AgenticTrace(BaseModel):
    """
    A single step recorded during the ReAct loop.

    The Streamlit UI renders a list of these inside the
    "Agentic Trace / Logs" expander.  Keys are chosen to map directly
    onto UI labels — do **not** rename them without updating
    ``ui/app.py``.

    Attributes
    ----------
    iteration : int
        Zero-based iteration counter inside the reflection loop.
    step_type : str
        One of ``"INIT"``, ``"REFLECT"``, ``"PLAN"``, ``"FILTER"``,
        ``"REFINE"``, ``"TERMINATE"``.
    adapter_state : str
        ``"ON"`` or ``"OFF"`` — which LoRA adapter state was active.
    thought : str
        The model's internal chain-of-thought at this step.
    action : str
        The action label executed (e.g. ``"extract_text"``,
        ``"diagnose_errors"``, ``"guided_refinement"``).
    observation : str
        The text output produced by the action.
    memory_snapshot : list[str]
        Snapshot of :attr:`AgenticMemory.past_reflections` at this step.
    """
    iteration       : int
    step_type       : str
    adapter_state   : str  # "ON" | "OFF"
    thought         : str  = ""
    action          : str  = ""
    observation     : str  = ""
    memory_snapshot : list[str] = Field(default_factory=list)


class AgenticMemory(BaseModel):
    """
    Mutable state maintained across iterations of the reflection loop.

    Corresponds to the *Reflection Memory Store* M_i formalised in
    §3.3 of the OCR-Agent paper.

    Attributes
    ----------
    iteration_count : int
        Current iteration index (0 = initial extraction).
    past_transcriptions : list[str]
        Chronological list of all raw transcription outputs.
    past_reflections : list[str]
        Chronological list of all reflection / diagnosis strings.
    past_plans : list[str]
        Chronological list of all *feasible* correction plans P_feas.
    """
    iteration_count     : int       = 0
    past_transcriptions : list[str] = Field(default_factory=list)
    past_reflections    : list[str] = Field(default_factory=list)
    past_plans          : list[str] = Field(default_factory=list)

    def record_iteration(
        self,
        transcription : str,
        reflection    : str,
        plan          : str,
    ) -> None:
        """Append a completed iteration to all history lists and bump the counter."""
        self.past_transcriptions.append(transcription)
        self.past_reflections.append(reflection)
        self.past_plans.append(plan)
        self.iteration_count += 1

    def has_stagnated(self) -> bool:
        """
        Return ``True`` if the last two transcriptions are identical,
        indicating refinement stagnation (§3.3 of OCR-Agent paper).
        """
        if len(self.past_transcriptions) < 2:
            return False
        return self.past_transcriptions[-1] == self.past_transcriptions[-2]


class OCRResult(BaseModel):
    """
    Final output of a complete pipeline run for one :class:`DocumentPage`.

    Attributes
    ----------
    page_id : str
        Mirrors :attr:`DocumentPage.page_id`.
    raw_text : str
        The final extracted / refined transcription string.
    execution_mode : ExecutionMode
        Which pipeline mode produced this result.
    confidence_score : float | None
        Optional model-reported confidence (0–1).  ``None`` if the VLM
        does not provide a logit-level score.
    agentic_trace : list[AgenticTrace]
        Ordered list of every step taken during the ReAct loop.
        Empty for one-shot modes.
    """
    page_id          : str
    raw_text         : str
    execution_mode   : ExecutionMode
    confidence_score : float | None          = None
    agentic_trace    : list[AgenticTrace]    = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """
    Per-page evaluation metrics produced by
    :class:`~src.evaluation.evaluator.Evaluator`.

    Attributes
    ----------
    page_id : str
        Mirrors :attr:`DocumentPage.page_id`.
    cer_score : float
        Character Error Rate (0–1).  Lower is better.
    wer_score : float
        Word Error Rate (0–1).  Lower is better.
    frequent_errors : dict[str, dict[str, int]]
        Nested mapping ``{ground_truth_char: {predicted_char: count}}``.
        Visualises the confusion matrix for common substitutions.
    error_heatmap_path : str | None
        Path to the generated error-overlay image, or ``None`` if
        visualisation was not requested.
    """
    page_id             : str
    cer_score           : float
    wer_score           : float
    frequent_errors     : dict[str, dict[str, int]] = Field(default_factory=dict)
    error_heatmap_path  : str | None                = None


class ParsedTranscription(BaseModel):
    """
    Structured output parsed from the raw VLM response.

    The model's response may contain a reasoning preamble, markdown
    formatting, or thinking tags before the actual transcription.
    This model isolates the clean transcription text only.
    """
    transcription: str = Field(
        description=(
            "The clean manuscript transcription only. "
            "No preamble, no explanations, no markdown, no thinking tags."
        )
    )
