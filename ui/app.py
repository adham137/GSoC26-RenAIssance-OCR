"""
ui/app.py
=========
Streamlit GUI for the Agentic OCR Framework.

Layout
------
┌─────────────────────────────────────────────────────────────────┐
│  Sidebar                   │  Main Area                         │
│  ─────────────────         │  ─────────────────────────────────  │
│  [Execution Mode]          │  [Left col]      [Right col]        │
│  [Adapter Path]            │  Manuscript img  Final OCR text     │
│  [Max Iterations]          │                                     │
│  [Upload PDF]              │  ─────────────────────────────────  │
│  [Run Pipeline]            │  ▼ Agentic Trace / Logs (expander)  │
│                            │    iter N | REFLECT | adapter=OFF   │
└─────────────────────────────────────────────────────────────────┘

Running
-------
    streamlit run ui/app.py

Requirements
------------
All heavy ML imports are inside the ``_run_pipeline()`` function so the
UI renders immediately even before model weights are loaded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# sys.path fix — must happen BEFORE any 'from src.*' import.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONPATH", str(ROOT))

from src.models import ExecutionMode

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic OCR Framework",
    page_icon="📜",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    execution_mode_label = st.selectbox(
        label="Execution Mode",
        options=[m.value for m in ExecutionMode],
        index=3,
        help=(
            "Base_OneShot: single-pass, no adapters.\n"
            "Base_ReAct: reflection loop, no adapters.\n"
            "Adapter_OneShot: single-pass with LoRA.\n"
            "Adapter_ReAct: full pipeline (recommended)."
        ),
    )
    execution_mode = ExecutionMode(execution_mode_label)

    _adapter_required = execution_mode in (
        ExecutionMode.ADAPTER_ONE_SHOT,
        ExecutionMode.ADAPTER_REACT,
    )
    adapter_path_input = st.text_input(
        label="LoRA Adapter Path"
        + ("" if _adapter_required else " (not used in base modes)"),
        value="" if not _adapter_required else "models/lora_manuscript_v1",
        placeholder="models/lora_manuscript_v1",
        disabled=not _adapter_required,
    )

    max_iterations = st.slider(
        label="Max Reflection Iterations",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        disabled=execution_mode
        in (ExecutionMode.BASE_ONE_SHOT, ExecutionMode.ADAPTER_ONE_SHOT),
    )

    load_in_4bit = st.checkbox(
        label="Load model in 4-bit (BnB)",
        value=True,
        help="Reduces VRAM usage. Recommended for consumer GPUs.",
    )

    use_sdpa = st.checkbox(
        label="Use SDPA attention (faster)",
        value=True,
        help="Scaled Dot-Product Attention for faster inference.",
    )

    verbose = st.checkbox(
        label="Verbose generation logging",
        value=False,
        help="Show token generation stats in the Generation Logs window.",
    )

    st.markdown("---")
    st.subheader("📂 Input Document")
    uploaded_file = st.file_uploader(
        label="Upload PDF Manuscript",
        type=["pdf"],
    )

    ground_truth_file = st.file_uploader(
        label="Upload Ground Truth (optional)",
        type=["txt"],
    )

    st.markdown("---")
    run_button = st.button(
        label="🚀 Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

# ---------------------------------------------------------------------------
# Main area — Header
# ---------------------------------------------------------------------------

st.title("📜 Agentic OCR Framework")
st.caption(
    "Iterative self-correction pipeline for 17th-century Spanish manuscripts "
    "via Capability Reflection & Memory Reflection (OCR-Agent, 2026)."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = []
if "eval_reports" not in st.session_state:
    st.session_state.eval_reports = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _render_trace_step(trace) -> None:
    """Render a single AgenticTrace entry inside the Streamlit expander."""
    STEP_CONFIG = {
        "INIT": ("🔍", st.info, "blue"),
        "REFLECT": ("🪞", st.warning, "orange"),
        "FILTER": ("🔎", st.info, "grey"),
        "REFINE": ("✏️", st.success, "green"),
        "TERMINATE": ("🛑", st.error, "red"),
    }
    icon, box_fn, _ = STEP_CONFIG.get(trace.step_type, ("❓", st.info, "grey"))

    adapter_badge = (
        "🟢 **Adapter: ON**" if trace.adapter_state == "ON" else "🔴 **Adapter: OFF**"
    )

    header = (
        f"{icon} **Iter {trace.iteration} | {trace.step_type}** "
        f"— `{trace.action}` — {adapter_badge}"
    )

    with st.container():
        box_fn(header)

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            if trace.thought:
                with st.expander("💭 Thought / Prompt (truncated)", expanded=False):
                    st.code(
                        trace.thought[:800] + ("…" if len(trace.thought) > 800 else ""),
                        language=None,
                    )

        with sub_col2:
            if trace.observation:
                with st.expander("👁️ Observation / Output", expanded=True):
                    st.text(
                        trace.observation[:600]
                        + ("…" if len(trace.observation) > 600 else "")
                    )

        if trace.memory_snapshot:
            with st.expander(
                f"🧠 Memory Snapshot ({len(trace.memory_snapshot)} entries)",
                expanded=False,
            ):
                for i, mem_entry in enumerate(trace.memory_snapshot):
                    st.markdown(
                        f"**M[{i}]:** {mem_entry[:300]}{'…' if len(mem_entry) > 300 else ''}"
                    )

        st.markdown("")


def _run_pipeline(
    uploaded_file,
    execution_mode: ExecutionMode,
    adapter_path: str,
    max_iters: int,
    load_4bit: bool,
    ground_truth_file,
    verbose: bool = False,
    use_sdpa: bool = True,
) -> None:
    """
    Lazy-import and execute the full pipeline. Called only when the user
    clicks "Run Pipeline."
    """
    import logging
    import tempfile
    from io import StringIO

    from src.ingestion.pdf_handler import PDFHandler
    from src.logger.trace_logger import TraceLogger
    from src.model_engine.model_executor import ModelExecutor
    from src.orchestrator.agentic_orchestrator import AgenticOrchestrator
    from src.prompt_manager.prompt_registry import PromptRegistry

    # ── Create log window for verbose output ─────────────────────────────
    log_container = st.empty()
    log_output = StringIO()

    class StreamlitLogHandler(logging.Handler):
        """Custom logging handler that writes to Streamlit UI."""
        def emit(self, record):
            msg = self.format(record)
            log_output.write(msg + "\n")
            with log_container:
                with st.expander("📋 Generation Logs", expanded=True):
                    st.code(
                        log_output.getvalue() or "Waiting for generation...",
                        language="log",
                        line_numbers=True,
                    )

    # Set up handler for model_executor logger
    if verbose:
        log_handler = StreamlitLogHandler()
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        model_logger = logging.getLogger("src.model_engine.model_executor")
        model_logger.addHandler(log_handler)
        model_logger.setLevel(logging.INFO)

    # ── Save uploaded PDF to a temp file ─────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_pdf_path = tmp.name

    progress_bar = st.progress(0, text="Initialising pipeline…")

    # ── Step 1: Ingest PDF ───────────────────────────────────────────────
    progress_bar.progress(10, text="📄 Splitting PDF into pages…")
    handler = PDFHandler(output_dir=str(ROOT / "data" / "output_images"), dpi=300)
    try:
        pages = handler.load_pdf(tmp_pdf_path)
    except NotImplementedError:
        st.error(
            "PDFHandler is not yet implemented. See `src/ingestion/pdf_handler.py`."
        )
        progress_bar.empty()
        return

    # ── Step 2: Load model ───────────────────────────────────────────────
    progress_bar.progress(25, text="🤖 Loading VLM (this may take a minute)…")

    _base_modes = (ExecutionMode.BASE_ONE_SHOT, ExecutionMode.BASE_REACT)
    resolved_adapter_path = (
        None
        if execution_mode in _base_modes or not adapter_path.strip()
        else adapter_path.strip()
    )

    executor = ModelExecutor(
        adapter_path=resolved_adapter_path,
        load_in_4bit=load_4bit,
        use_sdpa=use_sdpa,
        verbose=verbose,
    )
    try:
        executor.load_model()
    except NotImplementedError:
        st.error(
            "ModelExecutor.load_model() is not yet implemented."
        )
        progress_bar.empty()
        return
    except RuntimeError as exc:
        st.error(f"Model failed to load: {exc}")
        progress_bar.empty()
        return

    registry = PromptRegistry(prompts_dir=str(ROOT / "prompts"))

    # ── Step 3: Process each page ────────────────────────────────────────
    results = []
    n_pages = len(pages)

    for idx, page in enumerate(pages):
        progress_bar.progress(
            int(25 + 70 * (idx / n_pages)),
            text=f"🔍 Processing page {idx + 1}/{n_pages}…",
        )
        trace_logger = TraceLogger(logs_dir=str(ROOT / "logs"), page_id=page.page_id)
        orchestrator = AgenticOrchestrator(
            executor=executor,
            registry=registry,
            trace_logger=trace_logger,
            max_iterations=max_iters,
            execution_mode=execution_mode,
        )
        result = orchestrator.run(page)
        results.append(result)

    progress_bar.progress(100, text="✅ Pipeline complete!")
    progress_bar.empty()

    # Clean up log handler
    if verbose:
        model_logger.removeHandler(log_handler)

    st.session_state.ocr_results = results
    st.session_state.pipeline_done = True
    st.session_state.current_page = 0
    st.rerun()


# ---------------------------------------------------------------------------
# Trigger pipeline
# ---------------------------------------------------------------------------

if run_button and uploaded_file is not None:
    _run_pipeline(
        uploaded_file=uploaded_file,
        execution_mode=execution_mode,
        adapter_path=adapter_path_input,
        max_iters=max_iterations,
        load_4bit=load_in_4bit,
        ground_truth_file=ground_truth_file,
        verbose=verbose,
        use_sdpa=use_sdpa,
    )

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if not st.session_state.pipeline_done or not st.session_state.ocr_results:
    st.info(
        "👈 Upload a PDF and click **Run Pipeline** to begin.\n\n"
        "This framework implements the **OCR-Agent** self-correction loop:\n"
        "Capability Reflection → Memory Reflection → Guided Refinement.",
        icon="📜",
    )
    st.stop()

# ── Page navigator ───────────────────────────────────────────────────────
results = st.session_state.ocr_results
n_pages = len(results)
current_i = st.session_state.current_page

if n_pages > 1:
    col_prev, col_counter, col_next = st.columns([1, 3, 1])
    with col_prev:
        if st.button("◀ Previous", disabled=current_i == 0):
            st.session_state.current_page -= 1
            st.rerun()
    with col_counter:
        st.markdown(
            f"<div style='text-align:center;font-size:16px'>Page {current_i + 1} / {n_pages}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("Next ▶", disabled=current_i == n_pages - 1):
            st.session_state.current_page += 1
            st.rerun()
    st.markdown("---")

result = results[st.session_state.current_page]

# ── Main two-column layout ───────────────────────────────────────────────
col_image, col_text = st.columns([1, 1], gap="large")

with col_image:
    st.subheader("🖼️ Manuscript Image")
    if Path(result.page_id).exists():
        st.image(result.page_id, use_column_width=True)
    else:
        candidate = Path("data/output_images") / f"{result.page_id}.png"
        if candidate.exists():
            st.image(str(candidate), use_column_width=True)
        else:
            st.warning(f"Image not found for page: `{result.page_id}`")

    st.caption(
        f"**Mode:** `{result.execution_mode.value}` | "
        f"**Iterations:** {len([t for t in result.agentic_trace if t.step_type == 'REFINE'])} | "
        f"**Confidence:** {result.confidence_score if result.confidence_score else 'N/A'}"
    )

with col_text:
    st.subheader("📝 Extracted Transcription")
    st.text_area(
        label="Final OCR Output",
        value=result.raw_text,
        height=450,
        label_visibility="collapsed",
    )
    if result.raw_text:
        st.download_button(
            label="⬇️ Download Transcription (.txt)",
            data=result.raw_text,
            file_name=f"{result.page_id}_transcription.txt",
            mime="text/plain",
        )

# ── Agentic Trace expander ───────────────────────────────────────────────
st.markdown("---")

with st.expander("🧠 Agentic Trace / Logs", expanded=False):
    if not result.agentic_trace:
        st.info("No trace available for one-shot modes.")
    else:
        n_reflect = len([t for t in result.agentic_trace if t.step_type == "REFLECT"])
        n_refine = len([t for t in result.agentic_trace if t.step_type == "REFINE"])
        terminated = any(t.step_type == "TERMINATE" for t in result.agentic_trace)

        st.markdown(
            f"**Total steps:** {len(result.agentic_trace)} | "
            f"**Reflections:** {n_reflect} | "
            f"**Refinements:** {n_refine} | "
            f"**Early termination:** {'⚠️ Yes (stagnation)' if terminated else '✅ No'}"
        )
        st.markdown("---")

        for trace in result.agentic_trace:
            _render_trace_step(trace)
