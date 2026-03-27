"""
batch_process_pdfs.py
======================
Batch processing script for running the Agentic OCR pipeline on multiple PDFs.

This script processes all PDFs in data/raw_pdfs and outputs results to a JSON file.
Features:
    - Incremental saving: Results are saved after each page is processed
    - Resume capability: Automatically resumes from last completed page if interrupted

Usage
-----
    python batch_process_pdfs.py

Configuration
-------------
The script uses the following settings (matching UI defaults):
    - Execution Mode: Base_ReAct
    - Max Reflection Iterations: 1
    - Max output tokens: 2048
    - Enable Lexical Post Processing: True
    - Ground Truth: Not required
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup paths and environment
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONPATH", str(ROOT))

import yaml

# Load config
config_path = ROOT / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Import pipeline components
from src.models import ExecutionMode
from src.evaluation.evaluator import Evaluator
from src.ingestion.pdf_handler import PDFHandler
from src.model_engine.backend_factory import create_backend
from src.orchestrator.agentic_orchestrator import AgenticOrchestrator
from src.postprocessing.lexical_processor import LexicalProcessor
from src.prompt_manager.prompt_registry import PromptRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "batch_processing.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = ROOT / "data" / "batch_processing_checkpoint.json"


def load_checkpoint() -> dict[str, Any]:
    """
    Load checkpoint from disk if it exists.

    Returns
    -------
    dict[str, Any]
        Checkpoint data with keys: 'completed_pdfs', 'current_pdf_index',
        'current_page_index', 'results'
    """
    if CHECKPOINT_PATH.exists():
        try:
            with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"Loaded checkpoint from {CHECKPOINT_PATH}")
            return checkpoint
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")

    return {
        "completed_pdfs": [],
        "current_pdf_index": 0,
        "current_page_index": 0,
        "results": [],
    }


def save_checkpoint(checkpoint: dict[str, Any]) -> None:
    """
    Save checkpoint to disk.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        Checkpoint data to save.
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    logger.debug(f"Checkpoint saved to {CHECKPOINT_PATH}")


def save_final_results(output_data: dict[str, Any], output_path: Path) -> None:
    """
    Save final results to JSON file.

    Parameters
    ----------
    output_data : dict[str, Any]
        Complete results data.
    output_path : Path
        Path to save results to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


def cleanup_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info(f"Cleaned up checkpoint file: {CHECKPOINT_PATH}")


def process_pdf(
    pdf_path: Path,
    executor: Any,
    registry: PromptRegistry,
    max_iterations: int,
    execution_mode: ExecutionMode,
    use_lexical_correction: bool,
    lexical_processor: LexicalProcessor | None,
    output_images_dir: Path,
    checkpoint: dict[str, Any],
    pdf_index: int,
) -> dict[str, Any]:
    """
    Process a single PDF and return results for all its pages.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to process.
    executor : Any
        The model backend executor.
    registry : PromptRegistry
        Prompt template registry.
    max_iterations : int
        Maximum reflection iterations.
    execution_mode : ExecutionMode
        Execution mode to use.
    use_lexical_correction : bool
        Whether to apply lexical post-processing.
    lexical_processor : LexicalProcessor | None
        Lexical processor instance if enabled.
    output_images_dir : Path
        Directory for output images.
    checkpoint : dict[str, Any]
        Checkpoint data for resume capability.
    pdf_index : int
        Index of this PDF in the list of PDFs.

    Returns
    -------
    dict[str, Any]
        Dictionary with PDF filename and list of page results.
    """
    logger.info(f"Processing PDF: {pdf_path.name}")

    # Check if this PDF was already completed
    if str(pdf_path) in checkpoint.get("completed_pdfs", []):
        logger.info(f"  Skipping already completed PDF: {pdf_path.name}")
        # Find the existing result in checkpoint
        for result in checkpoint.get("results", []):
            if result.get("pdf_path") == str(pdf_path):
                return result
        # If not found in results, still skip
        return {
            "pdf_filename": pdf_path.name,
            "pdf_path": str(pdf_path),
            "skipped": True,
            "pages": [],
        }

    # Ingest PDF
    handler = PDFHandler(output_dir=str(output_images_dir), dpi=300)
    pages = handler.load_pdf(pdf_path)

    logger.info(f"  Found {len(pages)} pages in {pdf_path.name}")

    # Determine starting page from checkpoint
    start_page = (
        checkpoint.get("current_page_index", 0)
        if pdf_index == checkpoint.get("current_pdf_index", 0)
        else 0
    )

    if start_page > 0:
        logger.info(f"  Resuming from page {start_page + 1}/{len(pages)}")

    # Get or create result entry for this PDF
    pdf_result = None
    for result in checkpoint.get("results", []):
        if result.get("pdf_path") == str(pdf_path):
            pdf_result = result
            break

    if pdf_result is None:
        pdf_result = {
            "pdf_filename": pdf_path.name,
            "pdf_path": str(pdf_path),
            "total_pages": len(pages),
            "pages": [],
        }
        checkpoint["results"].append(pdf_result)

    page_results = pdf_result["pages"]

    for idx in range(start_page, len(pages)):
        page = pages[idx]
        logger.info(f"  Processing page {idx + 1}/{len(pages)}...")

        # Create trace logger for this page
        from src.logger.trace_logger import TraceLogger

        trace_logger = TraceLogger(
            logs_dir=str(ROOT / "logs"),
            page_id=page.page_id,
        )

        # Create orchestrator
        orchestrator = AgenticOrchestrator(
            executor=executor,
            registry=registry,
            trace_logger=trace_logger,
            max_iterations=max_iterations,
            execution_mode=execution_mode,
        )

        # Run pipeline on this page
        result = orchestrator.run(page)

        # Apply lexical correction if enabled
        if use_lexical_correction and lexical_processor:
            result.raw_text = lexical_processor.process(result.raw_text)

        # Store result for this page
        page_results.append(
            {
                "page_id": result.page_id,
                "page_number": idx + 1,
                "predicted_text": result.raw_text,
            }
        )

        # Update checkpoint with progress
        checkpoint["current_pdf_index"] = pdf_index
        checkpoint["current_page_index"] = idx + 1
        save_checkpoint(checkpoint)

        logger.info(f"    Completed page {idx + 1}/{len(pages)}")

    # Mark PDF as completed
    if str(pdf_path) not in checkpoint["completed_pdfs"]:
        checkpoint["completed_pdfs"].append(str(pdf_path))
        checkpoint["current_page_index"] = 0  # Reset for next PDF
        save_checkpoint(checkpoint)

    logger.info(f"Completed: {pdf_path.name}\n")

    return pdf_result


def main():
    """Main entry point for batch processing."""
    # Configuration (matching UI settings from request)
    EXECUTION_MODE = ExecutionMode.BASE_REACT
    MAX_ITERATIONS = 1
    MAX_NEW_TOKENS = 2048
    USE_LEXICAL_CORRECTION = True

    # Paths
    raw_pdfs_dir = ROOT / "data" / "raw_pdfs"
    output_images_dir = ROOT / "data" / "output_images"
    output_json_path = ROOT / "data" / "batch_results.json"

    # Ensure directories exist
    output_images_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = sorted(list(raw_pdfs_dir.glob("*.pdf")))

    if not pdf_files:
        logger.error(f"No PDF files found in {raw_pdfs_dir}")
        return

    # Load checkpoint (for resume capability)
    checkpoint = load_checkpoint()

    # Check if we're resuming or starting fresh
    if checkpoint.get("completed_pdfs") or checkpoint.get("results"):
        completed_count = len(checkpoint.get("completed_pdfs", []))
        current_pdf_idx = checkpoint.get("current_pdf_index", 0)
        current_page_idx = checkpoint.get("current_page_index", 0)
        if current_pdf_idx < len(pdf_files):
            logger.info(
                f"Resuming from PDF {current_pdf_idx + 1}/{len(pdf_files)}, page {current_page_idx + 1}"
            )
            logger.info(f"Previously completed PDFs: {completed_count}")
        else:
            logger.info(
                f"All PDFs already completed. Re-running to generate final output."
            )
    else:
        logger.info("Starting fresh batch processing run.")

    logger.info(f"Found {len(pdf_files)} PDF(s) to process")
    logger.info(f"Configuration:")
    logger.info(f"  - Execution Mode: {EXECUTION_MODE.value}")
    logger.info(f"  - Max Iterations: {MAX_ITERATIONS}")
    logger.info(f"  - Max Tokens: {MAX_NEW_TOKENS}")
    logger.info(f"  - Lexical Correction: {USE_LEXICAL_CORRECTION}")

    # Initialize shared components
    logger.info("Initializing backend...")
    backend = create_backend(config)

    # Override max_tokens for OpenRouter backend
    if config.get("backend") == "openrouter" and hasattr(backend, "_max_tokens"):
        backend._max_tokens = MAX_NEW_TOKENS
        logger.info(f"Set OpenRouter max_tokens to {MAX_NEW_TOKENS}")

    registry = PromptRegistry(prompts_dir=str(ROOT / "prompts"))
    lexical_processor = LexicalProcessor() if USE_LEXICAL_CORRECTION else None

    # Process all PDFs
    for pdf_index, pdf_path in enumerate(pdf_files):
        try:
            # Skip already completed PDFs
            if str(pdf_path) in checkpoint.get("completed_pdfs", []):
                continue

            result = process_pdf(
                pdf_path=pdf_path,
                executor=backend,
                registry=registry,
                max_iterations=MAX_ITERATIONS,
                execution_mode=EXECUTION_MODE,
                use_lexical_correction=USE_LEXICAL_CORRECTION,
                lexical_processor=lexical_processor,
                output_images_dir=output_images_dir,
                checkpoint=checkpoint,
                pdf_index=pdf_index,
            )
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
            # Record error in checkpoint
            error_result = {
                "pdf_filename": pdf_path.name,
                "pdf_path": str(pdf_path),
                "error": str(e),
                "pages": [],
            }
            # Check if already in results
            found = False
            for r in checkpoint.get("results", []):
                if r.get("pdf_path") == str(pdf_path):
                    found = True
                    break
            if not found:
                checkpoint["results"].append(error_result)
            save_checkpoint(checkpoint)

    # Build final output from checkpoint results
    output_data = {
        "configuration": {
            "execution_mode": EXECUTION_MODE.value,
            "max_iterations": MAX_ITERATIONS,
            "max_tokens": MAX_NEW_TOKENS,
            "lexical_correction_enabled": USE_LEXICAL_CORRECTION,
        },
        "total_pdfs": len(pdf_files),
        "results": checkpoint.get("results", []),
    }

    # Save final results
    save_final_results(output_data, output_json_path)

    # Cleanup checkpoint on successful completion
    cleanup_checkpoint()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Batch processing complete!")
    logger.info(f"Results saved to: {output_json_path}")
    logger.info(f"Total PDFs processed: {len(checkpoint.get('results', []))}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
