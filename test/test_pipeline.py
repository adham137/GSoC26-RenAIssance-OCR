"""
tests/test_pipeline.py
=======================
Pytest test suite for the Agentic OCR Framework.

Run with:
    pytest tests/ -v

Test categories
---------------
* ``test_memory_reflection``     — AgenticMemory state machine correctness
* ``test_dynamic_lora_switching``— Ensures adapter toggles correctly per step
* ``test_pdf_to_image_conversion``— PDFHandler chunking and image validity
* ``test_evaluation_metrics``    — CER / WER calculation against known values

Mocking Strategy
----------------
* Model inference is fully mocked — tests never require a GPU or model weights.
* ``peft`` / ``transformers`` are mocked at the class level so tests run in CI.
* The ``ModelExecutor`` is replaced with a ``MockModelExecutor`` that records
  adapter state at each call, enabling precise assertion of the switching rule.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.models import (
    AgenticMemory,
    AgenticTrace,
    DocumentPage,
    ExecutionMode,
    OCRResult,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_page(tmp_path: Path) -> DocumentPage:
    """
    Create a minimal DocumentPage pointing at a tiny dummy PNG.

    The PNG is a 10×10 white square — valid enough for Pillow to open.
    """
    from PIL import Image
    img_path = tmp_path / "sample_page.png"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(str(img_path))
    return DocumentPage(
        page_id    = "test_page_001",
        image_path = str(img_path),
        metadata   = {"source_year": 1640, "page_number": 1},
    )


@pytest.fixture
def fresh_memory() -> AgenticMemory:
    """Return a blank AgenticMemory at iteration 0."""
    return AgenticMemory()


@pytest.fixture
def mock_executor() -> MagicMock:
    """
    A MagicMock that mimics ModelExecutor's public interface.

    Tracks calls to set_adapter() / disable_adapter() and returns
    predictable text strings from inference methods.
    """
    executor                        = MagicMock()
    executor.adapter_state          = "OFF"       # initial state

    def _set_adapter():
        executor.adapter_state = "ON"

    def _disable_adapter():
        executor.adapter_state = "OFF"

    executor.set_adapter.side_effect    = _set_adapter
    executor.disable_adapter.side_effect = _disable_adapter

    # Inference stubs
    executor.extract_text.return_value       = "Initial transcription attempt."
    executor.diagnose_errors.return_value    = "Error: misread long-s as f. Plan: focus on descenders."
    executor.filter_plan.return_value        = "Plan: focus on descenders."  # feasible subset
    executor.guided_refinement.return_value  = "Refined transcription text."

    # _parse_transcription stub — returns input unchanged for testing
    executor._parse_transcription.side_effect = lambda x: x if isinstance(x, str) else str(x)

    return executor


@pytest.fixture
def mock_registry() -> MagicMock:
    """A PromptRegistry mock that returns the prompt_key as its rendered output."""
    registry        = MagicMock()
    registry.render = MagicMock(side_effect=lambda key, **_: f"[PROMPT:{key}]")
    return registry


@pytest.fixture
def mock_trace_logger() -> MagicMock:
    """A silent TraceLogger mock."""
    return MagicMock()


# ===========================================================================
# Test 1: Memory Reflection
# ===========================================================================

class TestMemoryReflection:
    """
    Validates the AgenticMemory state machine defined in src/models.py
    and its integration with the orchestrator loop.
    """

    def test_initial_state_is_empty(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: freshly constructed AgenticMemory.
        Assert: iteration_count=0 and all history lists are empty.
        """
        assert fresh_memory.iteration_count      == 0
        assert fresh_memory.past_transcriptions  == []
        assert fresh_memory.past_reflections     == []
        assert fresh_memory.past_plans           == []

    def test_record_iteration_appends_to_all_lists(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: call record_iteration() once with known strings.
        Assert: all three lists have exactly one entry and iteration_count == 1.
        """
        fresh_memory.record_iteration(
            transcription = "Thi5 is a te5t.",
            reflection    = "Digit 5 is misread from long-s glyph.",
            plan          = "Re-examine descender strokes.",
        )
        assert fresh_memory.iteration_count         == 1
        assert len(fresh_memory.past_transcriptions) == 1
        assert len(fresh_memory.past_reflections)    == 1
        assert len(fresh_memory.past_plans)          == 1
        assert fresh_memory.past_transcriptions[0]  == "Thi5 is a te5t."

    def test_memory_prevents_identical_sequential_outputs(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: record two iterations with the same transcription.
        Assert: has_stagnated() returns True (refinement stagnation detected).

        This mirrors the §3.3 paper condition where the loop should
        terminate rather than produce redundant corrections.
        """
        text = "Same text output unchanged."
        fresh_memory.record_iteration(text, "reflection_1", "plan_1")
        fresh_memory.record_iteration(text, "reflection_2", "plan_2")  # identical transcription
        assert fresh_memory.has_stagnated() is True

    def test_no_stagnation_when_outputs_differ(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: record two iterations with different transcriptions.
        Assert: has_stagnated() returns False — loop should continue.
        """
        fresh_memory.record_iteration("First attempt.", "reflection_1", "plan_1")
        fresh_memory.record_iteration("Second attempt, improved.", "reflection_2", "plan_2")
        assert fresh_memory.has_stagnated() is False

    def test_no_stagnation_with_single_entry(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: only one iteration recorded.
        Assert: has_stagnated() returns False (need ≥ 2 to compare).
        """
        fresh_memory.record_iteration("Only one entry.", "reflection_1", "plan_1")
        assert fresh_memory.has_stagnated() is False

    def test_memory_history_grows_monotonically(self, fresh_memory: AgenticMemory) -> None:
        """
        Setup: run 3 iterations with distinct content.
        Assert: all list lengths equal 3 and iteration_count == 3.
        """
        for i in range(3):
            fresh_memory.record_iteration(
                transcription = f"Transcription iteration {i}",
                reflection    = f"Reflection {i}",
                plan          = f"Plan {i}",
            )
        assert fresh_memory.iteration_count == 3
        assert len(fresh_memory.past_transcriptions) == 3

    def test_orchestrator_terminates_on_stagnation(
        self,
        sample_page   : DocumentPage,
        mock_executor : MagicMock,
        mock_registry : MagicMock,
        mock_trace_logger: MagicMock,
    ) -> None:
        """
        Setup: configure mock executor so guided_refinement always returns
               the same string, triggering stagnation after iteration 1.
        Assert: the returned OCRResult contains a TERMINATE trace entry.

        This is an integration test for the orchestrator's stagnation guard.
        """
        from src.orchestrator.agentic_orchestrator import AgenticOrchestrator

        FIXED_TEXT = "Identical text — will stagnate."
        mock_executor.extract_text.return_value       = FIXED_TEXT
        mock_executor.guided_refinement.return_value  = FIXED_TEXT

        orch = AgenticOrchestrator(
            executor       = mock_executor,
            registry       = mock_registry,
            trace_logger   = mock_trace_logger,
            max_iterations = 3,
            execution_mode = ExecutionMode.BASE_REACT,
        )
        result = orch.run(sample_page)

        step_types = [t.step_type for t in result.agentic_trace]
        assert "TERMINATE" in step_types, (
            "Orchestrator should emit a TERMINATE trace when stagnation is detected."
        )


# ===========================================================================
# Test 2: Dynamic LoRA Switching
# ===========================================================================

class TestDynamicLoRASwitching:
    """
    Verifies that the ModelExecutor adapter is toggled according to the
    strict rule documented in src/model_engine/model_executor.py and
    src/orchestrator/agentic_orchestrator.py:

        extract_text / guided_refinement → adapter ON
        diagnose_errors / filter_plan    → adapter OFF
    """

    def test_adapter_is_off_during_diagnose_errors(
        self,
        sample_page   : DocumentPage,
        mock_executor : MagicMock,
        mock_registry : MagicMock,
        mock_trace_logger: MagicMock,
    ) -> None:
        """
        Setup: run one ADAPTER_REACT iteration.
        Assert: every ``diagnose_errors`` call occurs while adapter_state == "OFF".

        We capture the adapter_state inside a side-effect closure to record
        the state *at call time*, not just before/after.
        """
        from src.orchestrator.agentic_orchestrator import AgenticOrchestrator

        adapter_state_at_diagnose: list[str] = []

        original_diagnose = mock_executor.diagnose_errors.side_effect

        def recording_diagnose(*args, **kwargs):
            adapter_state_at_diagnose.append(mock_executor.adapter_state)
            return "Error plan from diagnose."

        mock_executor.diagnose_errors.side_effect = recording_diagnose

        orch = AgenticOrchestrator(
            executor       = mock_executor,
            registry       = mock_registry,
            trace_logger   = mock_trace_logger,
            max_iterations = 1,
            execution_mode = ExecutionMode.ADAPTER_REACT,
        )
        orch.run(sample_page)

        for state in adapter_state_at_diagnose:
            assert state == "OFF", (
                f"diagnose_errors() was called with adapter_state='{state}'. "
                "Adapter MUST be OFF during Capability Reflection."
            )

    def test_adapter_is_on_during_extract_text(
        self,
        sample_page   : DocumentPage,
        mock_executor : MagicMock,
        mock_registry : MagicMock,
        mock_trace_logger: MagicMock,
    ) -> None:
        """
        Setup: run ADAPTER_ONE_SHOT mode.
        Assert: extract_text() is called while adapter_state == "ON".
        """
        from src.orchestrator.agentic_orchestrator import AgenticOrchestrator

        adapter_state_at_extract: list[str] = []

        def recording_extract(*args, **kwargs):
            adapter_state_at_extract.append(mock_executor.adapter_state)
            return "Extracted text."

        mock_executor.extract_text.side_effect = recording_extract

        orch = AgenticOrchestrator(
            executor       = mock_executor,
            registry       = mock_registry,
            trace_logger   = mock_trace_logger,
            max_iterations = 1,
            execution_mode = ExecutionMode.ADAPTER_ONE_SHOT,
        )
        orch.run(sample_page)

        assert all(s == "ON" for s in adapter_state_at_extract), (
            "extract_text() MUST be called with adapter_state='ON'."
        )

    def test_adapter_is_on_during_guided_refinement(
        self,
        sample_page   : DocumentPage,
        mock_executor : MagicMock,
        mock_registry : MagicMock,
        mock_trace_logger: MagicMock,
    ) -> None:
        """
        Setup: run ADAPTER_REACT with max_iterations=1.
        Assert: guided_refinement() is called while adapter_state == "ON".
        """
        from src.orchestrator.agentic_orchestrator import AgenticOrchestrator

        adapter_state_at_refine: list[str] = []

        def recording_refine(*args, **kwargs):
            adapter_state_at_refine.append(mock_executor.adapter_state)
            return "Refined output."

        mock_executor.guided_refinement.side_effect = recording_refine

        orch = AgenticOrchestrator(
            executor       = mock_executor,
            registry       = mock_registry,
            trace_logger   = mock_trace_logger,
            max_iterations = 1,
            execution_mode = ExecutionMode.ADAPTER_REACT,
        )
        orch.run(sample_page)

        assert all(s == "ON" for s in adapter_state_at_refine), (
            "guided_refinement() MUST be called with adapter_state='ON'."
        )

    def test_set_adapter_raises_without_adapter_path(self) -> None:
        """
        Setup: construct ModelExecutor with adapter_path=None and call set_adapter().
        Assert: RuntimeError is raised (no adapter configured).

        NOTE: This test will pass only after set_adapter() is implemented.
        Marked xfail until then.
        """
        pytest.importorskip("peft", reason="peft not installed")

        from src.model_engine.model_executor import ModelExecutor
        executor = ModelExecutor(adapter_path=None)
        # NOTE: load_model() is not called intentionally — we test guard only
        with pytest.raises((RuntimeError, NotImplementedError)):
            executor.set_adapter()

    def test_adapter_toggle_count_for_n_iterations(
        self,
        sample_page   : DocumentPage,
        mock_executor : MagicMock,
        mock_registry : MagicMock,
        mock_trace_logger: MagicMock,
    ) -> None:
        """
        Setup: run ADAPTER_REACT with max_iterations=3.
        Assert: set_adapter() is called exactly N+1 times (1 initial + N refine),
                and disable_adapter() is called exactly N times (once per loop).

        N=3 → set_adapter: 4 calls, disable_adapter: 3 calls.
        """
        from src.orchestrator.agentic_orchestrator import AgenticOrchestrator

        # Prevent stagnation so all 3 iterations run
        call_count = [0]

        def unique_refinement(*args, **kwargs):
            call_count[0] += 1
            return f"Unique refinement output {call_count[0]}"

        mock_executor.guided_refinement.side_effect = unique_refinement

        orch = AgenticOrchestrator(
            executor       = mock_executor,
            registry       = mock_registry,
            trace_logger   = mock_trace_logger,
            max_iterations = 3,
            execution_mode = ExecutionMode.ADAPTER_REACT,
        )
        orch.run(sample_page)

        assert mock_executor.set_adapter.call_count    == 4, (
            "set_adapter should be called once for initial extraction + once per iteration."
        )
        assert mock_executor.disable_adapter.call_count == 3, (
            "disable_adapter should be called once per reflection phase."
        )


# ===========================================================================
# Test 3: PDF to Image Conversion
# ===========================================================================

class TestPDFToImageConversion:
    """
    Tests for src/ingestion/pdf_handler.PDFHandler.

    These tests use ``unittest.mock.patch`` to replace ``fitz`` (PyMuPDF)
    with a controlled mock, so no real PDF or GPU is needed.
    """

    def test_single_page_pdf_returns_one_document_page(self, tmp_path: Path) -> None:
        """
        Setup: mock fitz.open() to return a 1-page document.
        Assert: load_pdf() returns a list of length 1.
        Assert: the single DocumentPage has page_id ending in "_page_001".
        """
        # TODO: Implement once PDFHandler._split_and_rasterise() is done.
        # Mocking strategy:
        #
        #   with patch("src.ingestion.pdf_handler.fitz") as mock_fitz:
        #       mock_doc  = MagicMock()
        #       mock_page = MagicMock()
        #       mock_pixmap = MagicMock()
        #       mock_page.get_pixmap.return_value = mock_pixmap
        #       mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        #       mock_doc.__len__  = MagicMock(return_value=1)
        #       mock_fitz.open.return_value  = mock_doc
        #       mock_fitz.Matrix.return_value = MagicMock()
        #
        #       pdf_path = tmp_path / "test.pdf"
        #       pdf_path.write_bytes(b"%PDF-1.4 minimal")  # dummy file
        #
        #       handler = PDFHandler(output_dir=tmp_path / "out")
        #       pages   = handler.load_pdf(pdf_path)
        #
        #       assert len(pages) == 1
        #       assert pages[0].page_id.endswith("_page_001")
        pytest.skip("Requires PDFHandler implementation — see TODO in docstring.")

    def test_multi_page_pdf_returns_correct_page_count(self, tmp_path: Path) -> None:
        """
        Setup: mock fitz.open() to return a 5-page document.
        Assert: load_pdf() returns a list of exactly 5 DocumentPage objects.
        Assert: page_ids are sequentially numbered _page_001 through _page_005.
        """
        pytest.skip("Requires PDFHandler implementation.")

    def test_each_page_has_valid_image_path(self, tmp_path: Path) -> None:
        """
        Setup: mock fitz to write real (tiny) PNG files via pix.save().
        Assert: DocumentPage.image_path exists on disk and is non-empty.
        Assert: PIL.Image.open() can open the image without raising.
        """
        pytest.skip("Requires PDFHandler implementation.")

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        """
        Setup: pass a non-existent path to load_pdf().
        Assert: FileNotFoundError is raised (not a generic exception).
        """
        from src.ingestion.pdf_handler import PDFHandler
        handler = PDFHandler(output_dir=tmp_path / "out")
        with pytest.raises(FileNotFoundError):
            handler.load_pdf(tmp_path / "does_not_exist.pdf")

    def test_page_metadata_contains_page_number(self, tmp_path: Path) -> None:
        """
        Setup: mock a 3-page PDF.
        Assert: DocumentPage.metadata["page_number"] equals 1, 2, 3 respectively.
        """
        pytest.skip("Requires PDFHandler implementation.")

    def test_output_images_written_to_correct_directory(self, tmp_path: Path) -> None:
        """
        Setup: mock a 2-page PDF with a specific output_dir.
        Assert: both PNG files appear inside ``output_dir / <pdf_stem>/``.
        """
        pytest.skip("Requires PDFHandler implementation.")


# ===========================================================================
# Test 4: Evaluation Metrics
# ===========================================================================

class TestEvaluationMetrics:
    """
    Tests for src/evaluation/evaluator.Evaluator.

    Uses fixed GT/hypothesis pairs with known CER and WER to validate
    correctness of the metric implementations.
    """

    # Known test vectors -------------------------------------------------------
    # Perfect match
    PERFECT_GT   = "this is a test"
    PERFECT_HYP  = "this is a test"

    # One char substitution in a 14-char string → CER = 1/14 ≈ 0.0714
    ONE_CHAR_GT  = "this is a test"
    ONE_CHAR_HYP = "this is a tast"  # 'e' → 'a'

    # One word substitution in a 4-word string → WER = 1/4 = 0.25
    ONE_WORD_GT  = "this is a test"
    ONE_WORD_HYP = "this is a quiz"  # 'test' → 'quiz'

    # Complete mismatch
    MISMATCH_GT  = "abcde"
    MISMATCH_HYP = "vwxyz"

    # --------------------------------------------------------------------------

    def test_perfect_match_cer_is_zero(self) -> None:
        """
        Setup: hypothesis == reference.
        Assert: CER == 0.0 exactly.
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev = Evaluator(normalise=False)
        assert ev.compute_cer(self.PERFECT_HYP, self.PERFECT_GT) == pytest.approx(0.0)

    def test_perfect_match_wer_is_zero(self) -> None:
        """
        Setup: hypothesis == reference.
        Assert: WER == 0.0 exactly.
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev = Evaluator(normalise=False)
        assert ev.compute_wer(self.PERFECT_HYP, self.PERFECT_GT) == pytest.approx(0.0)

    def test_one_char_substitution_cer(self) -> None:
        """
        Setup: one character differs in a 14-character string (post-normalise).
        Assert: CER ≈ 1/14 ≈ 0.0714.
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev  = Evaluator(normalise=False)
        cer = ev.compute_cer(self.ONE_CHAR_HYP, self.ONE_CHAR_GT)
        assert cer == pytest.approx(1 / 14, abs=1e-3)

    def test_one_word_substitution_wer(self) -> None:
        """
        Setup: one word differs in a 4-word string.
        Assert: WER == 0.25 (1/4).
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev  = Evaluator(normalise=False)
        wer = ev.compute_wer(self.ONE_WORD_HYP, self.ONE_WORD_GT)
        assert wer == pytest.approx(0.25, abs=1e-3)

    def test_complete_mismatch_cer_above_zero(self) -> None:
        """
        Setup: completely different strings of equal length.
        Assert: CER > 0 (all characters are substitutions).
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev  = Evaluator(normalise=False)
        cer = ev.compute_cer(self.MISMATCH_HYP, self.MISMATCH_GT)
        assert cer > 0.0

    def test_normalisation_lowercases_before_scoring(self) -> None:
        """
        Setup: hypothesis and GT differ only in case.
        Assert: With normalise=True, CER == 0.0 (case-insensitive match).
        Assert: With normalise=False, CER > 0.0 (case-sensitive mismatch).
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator

        gt  = "This Is A Test"
        hyp = "this is a test"

        ev_norm = Evaluator(normalise=True)
        ev_raw  = Evaluator(normalise=False)

        assert ev_norm.compute_cer(hyp, gt) == pytest.approx(0.0)
        assert ev_raw.compute_cer(hyp, gt) > 0.0

    def test_evaluate_returns_report_with_correct_page_id(self) -> None:
        """
        Setup: call evaluate() with a known page_id.
        Assert: returned EvaluationReport.page_id matches the input.
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev     = Evaluator(normalise=False)
        report = ev.evaluate(
            page_id  = "test_page_042",
            ocr_text = self.PERFECT_HYP,
            gt_text  = self.PERFECT_GT,
        )
        assert report.page_id == "test_page_042"

    def test_evaluate_no_image_path_has_null_heatmap(self) -> None:
        """
        Setup: call evaluate() without providing an image_path.
        Assert: EvaluationReport.error_heatmap_path is None.
        """
        pytest.importorskip("jiwer", reason="jiwer not installed")
        from src.evaluation.evaluator import Evaluator
        ev     = Evaluator(normalise=False)
        report = ev.evaluate(
            page_id  = "test_page_001",
            ocr_text = self.PERFECT_HYP,
            gt_text  = self.PERFECT_GT,
            image_path = None,
        )
        assert report.error_heatmap_path is None

    def test_confusion_matrix_captures_known_substitution(self) -> None:
        """
        Setup: hypothesis has 'f' where GT has 's' (classic 17th-century
               long-s confusion).
        Assert: confusion_matrix["s"]["f"] >= 1.

        This validates the specific historical error pattern the system is
        designed to detect and track.
        """
        pytest.importorskip("Levenshtein", reason="python-Levenshtein not installed")
        from src.evaluation.evaluator import Evaluator
        ev     = Evaluator(normalise=False)
        matrix = ev.build_confusion_matrix(
            hypothesis = "the fouls sang",   # 's' → 'f' substitution
            reference  = "the souls sang",
        )
        assert "s" in matrix, "GT char 's' should appear in confusion matrix keys."
        assert "f" in matrix.get("s", {}), (
            "Substitution s→f should be recorded (historical long-s confusion)."
        )
        assert matrix["s"]["f"] >= 1
