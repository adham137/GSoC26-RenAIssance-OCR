"""
src/evaluation/run_eval.py
===========================
Command-line entry point for running the full pipeline and evaluation
without the Streamlit UI.

Usage
-----
    python -m src.evaluation.run_eval \\
        --pdf     data/raw_pdfs/manuscript_1640.pdf \\
        --gt      data/ground_truth/manuscript_1640_gt.txt \\
        --mode    Adapter_ReAct \\
        --adapter models/lora_manuscript_v1 \\
        --output  results/ \\
        --dpi     300 \\
        --iters   3

The GT file should contain one line of text per PDF page (in order).
If the GT file has fewer lines than pages, evaluation is skipped for
the remaining pages (transcription still runs).

Output
------
    results/<pdf_stem>_results.json   — per-page OCRResult + EvaluationReport
    results/<pdf_stem>_summary.txt    — mean CER / WER across all pages
    logs/                             — JSONL agentic trace files
    data/output_images/<pdf_stem>/    — rasterised page PNGs + heatmaps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import Evaluator
from src.ingestion.pdf_handler import PDFHandler
from src.logger.trace_logger import TraceLogger
from src.model_engine.model_executor import ModelExecutor
from src.models import ExecutionMode
from src.orchestrator.agentic_orchestrator import AgenticOrchestrator
from src.prompt_manager.prompt_registry import PromptRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agentic OCR Framework — CLI evaluation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pdf", required=True, help="Path to input PDF manuscript")
    p.add_argument(
        "--gt", default=None, help="Path to ground truth .txt file (one line per page)"
    )
    p.add_argument(
        "--mode",
        default="Adapter_ReAct",
        choices=[m.value for m in ExecutionMode],
        help="Execution mode",
    )
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    p.add_argument(
        "--output", default="results/", help="Output directory for JSON results"
    )
    p.add_argument("--dpi", type=int, default=300, help="PDF rasterisation DPI")
    p.add_argument("--iters", type=int, default=3, help="Max reflection iterations")
    p.add_argument("--no4bit", action="store_true", help="Disable 4-bit quantisation")
    return p.parse_args()


def load_ground_truth(gt_path: str | None) -> list[str]:
    """Load GT lines from a text file. Returns empty list if path is None."""
    if gt_path is None:
        return []
    lines = Path(gt_path).read_text(encoding="utf-8").splitlines()
    # Strip blank lines from the end but preserve internal blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Ingest PDF ────────────────────────────────────────────────────────
    logger.info("=== Phase 1: PDF Ingestion ===")
    handler = PDFHandler(output_dir="data/output_images", dpi=args.dpi)
    pages = handler.load_pdf(args.pdf)
    logger.info("Loaded %d pages.", len(pages))

    # ── 2. Load ground truth ─────────────────────────────────────────────────
    gt_lines = load_ground_truth(args.gt)
    if gt_lines:
        logger.info("Ground truth loaded: %d lines.", len(gt_lines))
    else:
        logger.info("No ground truth provided — evaluation metrics will be skipped.")

    # ── 3. Initialise model ──────────────────────────────────────────────────
    logger.info("=== Phase 2: Model Initialisation ===")
    executor = ModelExecutor(
        adapter_path=args.adapter,
        load_in_4bit=not args.no4bit,
    )
    executor.load_model()

    registry = PromptRegistry(prompts_dir=str(ROOT / "prompts"))
    evaluator = Evaluator(output_dir=str(output_dir / "heatmaps"))

    execution_mode = ExecutionMode(args.mode)

    # ── 4. Process each page ─────────────────────────────────────────────────
    logger.info("=== Phase 3: Pipeline Execution (%s) ===", execution_mode.value)

    all_results = []
    total_cer, total_wer, eval_count = 0.0, 0.0, 0

    for idx, page in enumerate(pages):
        logger.info("--- Page %d / %d  (%s) ---", idx + 1, len(pages), page.page_id)

        trace_logger = TraceLogger(logs_dir="logs/", page_id=page.page_id)
        orchestrator = AgenticOrchestrator(
            executor=executor,
            registry=registry,
            trace_logger=trace_logger,
            max_iterations=args.iters,
            execution_mode=execution_mode,
        )

        result = orchestrator.run(page)
        logger.info("Transcription complete (%d chars).", len(result.raw_text))

        # ── Evaluate if GT is available ───────────────────────────────────
        report_dict = None
        if idx < len(gt_lines):
            report = evaluator.evaluate(
                page_id=page.page_id,
                ocr_text=result.raw_text,
                gt_text=gt_lines[idx],
                image_path=page.image_path,
            )
            total_cer += report.cer_score
            total_wer += report.wer_score
            eval_count += 1
            report_dict = report.model_dump()
            logger.info("CER=%.4f  WER=%.4f", report.cer_score, report.wer_score)

        all_results.append(
            {
                "page_id": result.page_id,
                "mode": result.execution_mode.value,
                "raw_text": result.raw_text,
                "trace_steps": len(result.agentic_trace),
                "evaluation": report_dict,
            }
        )

    # ── 5. Write outputs ─────────────────────────────────────────────────────
    logger.info("=== Phase 4: Writing Results ===")
    pdf_stem = Path(args.pdf).stem
    results_path = output_dir / f"{pdf_stem}_results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Per-page results → %s", results_path)

    if eval_count > 0:
        mean_cer = total_cer / eval_count
        mean_wer = total_wer / eval_count
        summary = (
            f"=== Evaluation Summary ===\n"
            f"Document : {args.pdf}\n"
            f"Mode     : {args.mode}\n"
            f"Pages    : {eval_count}\n"
            f"Mean CER : {mean_cer:.4f}\n"
            f"Mean WER : {mean_wer:.4f}\n"
        )
        summary_path = output_dir / f"{pdf_stem}_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")
        print("\n" + summary)
        logger.info("Summary → %s", summary_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
