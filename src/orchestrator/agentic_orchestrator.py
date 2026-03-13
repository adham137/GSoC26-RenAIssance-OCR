"""
src/orchestrator/agentic_orchestrator.py
=========================================
Agentic Orchestrator — Reflection & Guided Refinement (System Design §2.4).

This module implements the iterative self-correction loop from the
OCR-Agent paper (Algorithm 1, §3).  It coordinates all other modules:
    * :class:`~src.model_engine.ModelExecutor` for inference
    * :class:`~src.prompt_manager.PromptRegistry` for template rendering
    * :class:`~src.logger.TraceLogger` for step recording

=============================================================================
CRITICAL: DYNAMIC LoRA SWITCHING LOGIC
=============================================================================
The adapter state transition inside each iteration MUST follow this pattern:

    ITERATION i
    ┌────────────────────────────────────────────────────────────────────┐
    │  1. executor.set_adapter()       ← adapter ON                     │
    │  2. text = executor.extract_text(...)   (if i == 0)               │
    │     OR                                                             │
    │     text = executor.guided_refinement(...)  (if i > 0)            │
    │                                                                    │
    │  3. executor.disable_adapter()   ← adapter OFF                    │
    │  4. raw_plan = executor.diagnose_errors(...)                       │
    │  5. feasible_plan = executor.filter_plan(...)                      │
    │                                                                    │
    │  (adapter stays OFF — loop back to step 1 for next iteration)     │
    └────────────────────────────────────────────────────────────────────┘

Breaking this pattern corrupts both the quality of planning (by biasing
it with manuscript-specific weights) and the quality of transcription
(by using un-specialised weights for glyph recognition).
=============================================================================

Termination Conditions
----------------------
The loop exits when **any** of the following is met:
    1. ``max_iterations`` is reached.
    2. :meth:`~src.models.AgenticMemory.has_stagnated` returns ``True``
       (two consecutive identical transcriptions).
    3. The feasible plan is empty (no actionable correction steps remain).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.models import (
    AgenticMemory,
    AgenticTrace,
    DocumentPage,
    ExecutionMode,
    OCRResult,
)

if TYPE_CHECKING:
    from src.logger.trace_logger import TraceLogger
    from src.model_engine.model_executor import ModelExecutor
    from src.prompt_manager.prompt_registry import PromptRegistry

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    """
    Drives the multi-turn Capability + Memory Reflection loop.

    Parameters
    ----------
    executor : ModelExecutor
        Pre-loaded inference engine (``load_model()`` already called).
    registry : PromptRegistry
        Prompt template store.
    trace_logger : TraceLogger
        Structured step recorder.
    max_iterations : int
        Hard cap on reflection rounds (default: 3, as per OCR-Agent paper).
    execution_mode : ExecutionMode
        Controls whether adapters are engaged and whether the loop runs.

    Examples
    --------
    >>> orchestrator = AgenticOrchestrator(
    ...     executor=executor,
    ...     registry=registry,
    ...     trace_logger=logger,
    ...     max_iterations=3,
    ...     execution_mode=ExecutionMode.ADAPTER_REACT,
    ... )
    >>> result = orchestrator.run(page)
    >>> print(result.raw_text[:100])
    """

    def __init__(
        self,
        executor        : "ModelExecutor",
        registry        : "PromptRegistry",
        trace_logger    : "TraceLogger",
        max_iterations  : int = 3,
        execution_mode  : ExecutionMode = ExecutionMode.ADAPTER_REACT,
    ) -> None:
        self.executor       = executor
        self.registry       = registry
        self.trace_logger   = trace_logger
        self.max_iterations = max_iterations
        self.execution_mode = execution_mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, page: DocumentPage) -> OCRResult:
        """
        Execute the full pipeline for a single manuscript page.

        Dispatches to the appropriate sub-method based on
        :attr:`execution_mode`.

        Parameters
        ----------
        page : DocumentPage
            The page to process.

        Returns
        -------
        OCRResult
            Final result with transcription, mode, and full agentic trace.
        """
        logger.info("Running orchestrator on page '%s' (mode=%s)", page.page_id, self.execution_mode)

        if self.execution_mode == ExecutionMode.BASE_ONE_SHOT:
            return self._run_one_shot(page, use_adapter=False)
        elif self.execution_mode == ExecutionMode.ADAPTER_ONE_SHOT:
            return self._run_one_shot(page, use_adapter=True)
        elif self.execution_mode == ExecutionMode.BASE_REACT:
            return self._run_react_loop(page, use_adapter=False)
        elif self.execution_mode == ExecutionMode.ADAPTER_REACT:
            return self._run_react_loop(page, use_adapter=True)
        else:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

    # ------------------------------------------------------------------
    # One-shot path
    # ------------------------------------------------------------------

    def _run_one_shot(self, page: DocumentPage, *, use_adapter: bool) -> OCRResult:
        """
        Single-pass extraction with no iterative refinement.

        Parameters
        ----------
        page : DocumentPage
            Target page.
        use_adapter : bool
            If ``True``, engage LoRA adapter before extraction.

        Returns
        -------
        OCRResult
            Result with an empty ``agentic_trace``.
        """
        if use_adapter:
            self.executor.set_adapter()

        prompt = self.registry.render("initial_extraction")
        text = self.executor.extract_text(page, prompt)
        text = self.executor._parse_transcription(text)

        trace_entry = AgenticTrace(
            iteration     = 0,
            step_type     = "INIT",
            adapter_state = self.executor.adapter_state,
            action        = "extract_text",
            observation   = text,
        )
        self.trace_logger.record(trace_entry)

        return OCRResult(
            page_id        = page.page_id,
            raw_text       = text,
            execution_mode = self.execution_mode,
            agentic_trace  = [trace_entry],
        )

    # ------------------------------------------------------------------
    # ReAct loop — the primary research pathway
    # ------------------------------------------------------------------

    def _run_react_loop(self, page: DocumentPage, *, use_adapter: bool) -> OCRResult:
        """
        Multi-turn Capability + Memory Reflection loop (Algorithm 1, §3).

        Each iteration follows the strict adapter-switching sequence:
            1. adapter ON  → extract / refine
            2. adapter OFF → reflect / plan / filter

        Parameters
        ----------
        page : DocumentPage
            Target page.
        use_adapter : bool
            If ``True``, engage LoRA adapter for read/refine steps.

        Returns
        -------
        OCRResult
            Result containing the final refined text and the full trace.
        """
        memory : AgenticMemory = AgenticMemory()
        traces : list[AgenticTrace] = []

        # ── Step 0: Initial extraction ──────────────────────────────────
        if use_adapter:
            self.executor.set_adapter()  # adapter ON for initial read

        init_prompt = self.registry.render("initial_extraction")
        current_text = self.executor.extract_text(page, init_prompt)

        init_trace = AgenticTrace(
            iteration     = 0,
            step_type     = "INIT",
            adapter_state = self.executor.adapter_state,
            action        = "extract_text",
            observation   = current_text,
        )
        traces.append(init_trace)
        self.trace_logger.record(init_trace)

        # ── Reflection loop ─────────────────────────────────────────────
        for i in range(1, self.max_iterations + 1):
            logger.debug("Starting iteration %d / %d", i, self.max_iterations)

            # ── Phase A: Capability Reflection — adapter OFF ─────────────
            if use_adapter:
                self.executor.disable_adapter()  # CRITICAL: adapter OFF for planning

            reflect_prompt = self.registry.render(
                "capability_reflection",
                current_text     = current_text,
                past_reflections = "\n---\n".join(memory.past_reflections),
                iteration        = str(i),
            )
            raw_plan = self.executor.diagnose_errors(page, current_text, reflect_prompt)

            reflect_trace = AgenticTrace(
                iteration       = i,
                step_type       = "REFLECT",
                adapter_state   = self.executor.adapter_state,
                action          = "diagnose_errors",
                thought         = reflect_prompt,
                observation     = raw_plan,
                memory_snapshot = list(memory.past_reflections),
            )
            traces.append(reflect_trace)
            self.trace_logger.record(reflect_trace)

            # ── Phase B: Filter plan — adapter stays OFF ──────────────────
            filter_prompt  = self.registry.render(
                "capability_filter",
                raw_plan = raw_plan,
            )
            feasible_plan  = self.executor.filter_plan(raw_plan, filter_prompt)

            filter_trace = AgenticTrace(
                iteration     = i,
                step_type     = "FILTER",
                adapter_state = self.executor.adapter_state,
                action        = "filter_plan",
                thought       = f"Filtering infeasible actions from plan (iteration {i})",
                observation   = feasible_plan,
            )
            traces.append(filter_trace)
            self.trace_logger.record(filter_trace)

            # ── Phase C: Memory Reflection — update M_i ───────────────────
            memory_prompt = self.registry.render(
                "memory_reflection",
                current_text     = current_text,
                feasible_plan    = feasible_plan,
                past_reflections = "\n---\n".join(memory.past_reflections),
            )

            # ── Phase D: Guided Refinement — adapter ON ───────────────────
            if use_adapter:
                self.executor.set_adapter()  # CRITICAL: adapter back ON for refining

            refine_prompt = self.registry.render(
                "guided_refinement",
                current_text     = current_text,
                feasible_plan    = feasible_plan,
                past_reflections = "\n---\n".join(memory.past_reflections),
            )
            refined_text = self.executor.guided_refinement(page, feasible_plan, refine_prompt)

            refine_trace = AgenticTrace(
                iteration       = i,
                step_type       = "REFINE",
                adapter_state   = self.executor.adapter_state,
                action          = "guided_refinement",
                thought         = refine_prompt,
                observation     = refined_text,
                memory_snapshot = list(memory.past_reflections),
            )
            traces.append(refine_trace)
            self.trace_logger.record(refine_trace)

            # ── Update memory with this iteration's results ───────────────
            memory.record_iteration(
                transcription = refined_text,
                reflection    = raw_plan,
                plan          = feasible_plan,
            )
            current_text = refined_text

            # ── Termination checks ────────────────────────────────────────
            if memory.has_stagnated():
                logger.info("Refinement stagnated at iteration %d — terminating loop.", i)
                term_trace = AgenticTrace(
                    iteration     = i,
                    step_type     = "TERMINATE",
                    adapter_state = self.executor.adapter_state,
                    action        = "stagnation_check",
                    thought       = "Consecutive identical outputs detected.",
                    observation   = "Loop terminated early due to stagnation.",
                )
                traces.append(term_trace)
                self.trace_logger.record(term_trace)
                break

            if not feasible_plan.strip():
                logger.info("Empty feasible plan at iteration %d — terminating loop.", i)
                break

        # Parse final transcription to remove any preamble/thinking
        current_text = self.executor._parse_transcription(current_text)

        return OCRResult(
            page_id        = page.page_id,
            raw_text       = current_text,
            execution_mode = self.execution_mode,
            agentic_trace  = traces,
        )
