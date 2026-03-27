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

from dotenv import load_dotenv
import streamlit as st

# ---------------------------------------------------------------------------
# Load .env file BEFORE anything else
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# sys.path fix — must happen BEFORE any 'from src.*' import.
# ---------------------------------------------------------------------------
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONPATH", str(ROOT))

import yaml

from src.models import ExecutionMode
from src.model_engine.adapter_downloader import AdapterDownloader, DownloadProgress

# ---------------------------------------------------------------------------
# Load config at module level
# ---------------------------------------------------------------------------

_config_path = ROOT / "config.yaml"
try:
    with open(_config_path) as _f:
        _app_config = yaml.safe_load(_f)
    _active_backend: str = _app_config.get("backend", "local")
except FileNotFoundError:
    _app_config = {}
    _active_backend = "local"

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

    # Backend indicator
    if _active_backend == "openrouter":
        _or_model = _app_config.get("openrouter", {}).get("model", "unknown model")
        st.success(f"🌐 **Backend: OpenRouter**\n\n`{_or_model}`")
        _api_key_set = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
        if not _api_key_set:
            st.warning(
                "⚠️ `OPENROUTER_API_KEY` not set. "
                "Pipeline will fail. Set it in your `.env` file.",
                icon="🔑",
            )
        else:
            st.caption("✅ API key detected")
    else:
        st.info("💻 **Backend: Local model**")
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

    # Adapter controls only shown for local backend
    if _active_backend == "local":
        _adapter_required = execution_mode in (
            ExecutionMode.ADAPTER_ONE_SHOT,
            ExecutionMode.ADAPTER_REACT,
        )

        # Get available adapters from models directory
        downloader = AdapterDownloader()
        available_adapters = downloader.list_available_adapters()
        adapter_options = [str(p) for p in available_adapters]

        adapter_path_input = st.text_input(
            label="LoRA Adapter Path"
            + ("" if _adapter_required else " (not used in base modes)"),
            value="" if not _adapter_required else "models/lora_manuscript_v1",
            placeholder="models/lora_manuscript_v1",
            disabled=not _adapter_required,
            help="Path to adapter directory or select from downloaded adapters below",
        )

        # Show available adapters if any exist
        if available_adapters and _adapter_required:
            with st.expander("📁 Available Downloaded Adapters", expanded=False):
                for adapter_path in available_adapters:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"`{adapter_path}`")
                    with col2:
                        if st.button(
                            "Use",
                            key=f"use_adapter_{adapter_path.name}",
                            type="secondary",
                        ):
                            st.session_state.selected_adapter_path = str(adapter_path)
                            st.rerun()

                if not available_adapters:
                    st.info(
                        "No adapters downloaded yet. Use the download section above."
                    )

        # Allow user to select a downloaded adapter
        if (
            hasattr(st.session_state, "selected_adapter_path")
            and st.session_state.selected_adapter_path
        ):
            adapter_path_input = st.session_state.selected_adapter_path
            st.info(f"✅ Using adapter: `{adapter_path_input}`")
            if st.button("Clear selection", key="clear_adapter_selection"):
                st.session_state.selected_adapter_path = None
                st.rerun()
    else:
        adapter_path_input = ""
        if execution_mode in (
            ExecutionMode.ADAPTER_ONE_SHOT,
            ExecutionMode.ADAPTER_REACT,
        ):
            st.info(
                "ℹ️ Adapter modes run as base modes when using the OpenRouter backend. "
                "The execution mode label is preserved in logs.",
                icon="🔄",
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

    max_new_tokens = st.slider(
        label="Max output tokens",
        min_value=256,
        max_value=2048,
        value=768,
        step=64,
        help=(
            "Maximum tokens the model can generate per page. "
            "A manuscript page is typically 400–800 tokens. "
            "Lower values = faster generation."
            if _active_backend == "local"
            else (
                "Maximum tokens the API will generate per page. "
                "Overrides the max_tokens value in config.yaml for this session. "
                "A manuscript page is typically 400–800 tokens."
            )
        ),
    )

    # Local-backend-only controls
    if _active_backend == "local":
        load_in_4bit = st.checkbox(
            label="Load model in 4-bit (BnB)",
            value=True,
            help="Reduces VRAM usage. Recommended for consumer GPUs.",
        )

        use_compile = st.checkbox(
            label="Compile model (torch.compile)",
            value=False,
            help=(
                "Compiles the model forward pass for ~20-40% faster generation. "
                "Adds ~2 min warm-up on first run. Best for processing many pages."
            ),
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
    else:
        load_in_4bit = False
        use_compile = False
        use_sdpa = False
        # For OpenRouter, always show API request/response logs
        verbose = True
        st.info(
            "ℹ️ API request/response logs are shown automatically for OpenRouter backend.",
            icon="📡",
        )

    use_lexical_correction = st.checkbox(
        label="Enable Lexical Post-Processing",
        value=False,
        help="Applies heuristic spacing corrections (e.g., splitting 'dela' to 'de la') to reduce Word Error Rate.",
    )

    st.markdown("---")

    # Adapter download section only for local backend
    if _active_backend == "local":
        # ── Adapter Download Section ─────────────────────────────────────────
        with st.expander("📥 Download Adapter from Hugging Face", expanded=False):
            st.caption("Download LoRA adapters from Hugging Face Hub")

            hf_repo_id = st.text_input(
                label="Hugging Face Repo ID",
                placeholder="username/my-adapter",
                key="hf_repo_id",
                help="Enter the Hugging Face repository ID (e.g., 'username/adapter-name')",
            )

            hf_token = st.text_input(
                label="Hugging Face Token (optional)",
                type="password",
                placeholder="hf_xxxxxxxxxxxxx",
                key="hf_token",
                help="Required for private repositories. Get your token from huggingface.co/settings/tokens",
            )

            adapter_local_name = st.text_input(
                label="Local Name (optional)",
                placeholder="my_adapter",
                key="adapter_local_name",
                help="Custom name for the downloaded adapter. Uses repo name if empty.",
            )

            # Download button with callback
            download_clicked = st.button(
                label="⬇️ Download",
                type="primary",
                use_container_width=True,
                disabled=not hf_repo_id.strip(),
                key="download_adapter_btn",
            )

            # Handle download immediately when button is clicked
            if download_clicked and hf_repo_id.strip():
                progress_bar = st.progress(0, text="Starting download...")
                status_text = st.empty()
                result_container = st.empty()

                try:
                    # Create progress callback
                    def on_progress(prog: DownloadProgress):
                        if prog.status == "downloading":
                            progress_bar.progress(
                                max(prog.percentage / 100, 0.01),
                                text=f"Downloading {prog.current_file or '...'} ({prog.downloaded_files}/{prog.total_files} files)",
                            )
                            status_text.caption(
                                f"📥 {prog.downloaded_files}/{prog.total_files} files"
                            )
                        elif prog.status == "validating":
                            progress_bar.progress(1.0, text="Validating adapter...")
                            status_text.caption("⏳ Validating downloaded files...")

                    # Download the adapter
                    downloader = AdapterDownloader()
                    adapter_path = downloader.download(
                        repo_id=hf_repo_id.strip(),
                        local_dir_name=adapter_local_name.strip()
                        if adapter_local_name.strip()
                        else None,
                        token=hf_token.strip() if hf_token.strip() else None,
                        progress_callback=on_progress,
                    )

                    # Success
                    progress_bar.empty()
                    status_text.empty()
                    result_container.success(
                        f"✅ Adapter downloaded to: `{adapter_path}`"
                    )

                    # Store in session state for later use
                    if str(adapter_path) not in st.session_state.downloaded_adapters:
                        st.session_state.downloaded_adapters.append(str(adapter_path))

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    result_container.error(f"❌ Download failed: {str(e)}")

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
    _api_key_missing = _active_backend == "openrouter" and not bool(
        os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    run_pipeline_button = st.button(
        label="🚀 Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or _api_key_missing,
    )
    if _api_key_missing:
        st.caption("❌ Cannot run: set `OPENROUTER_API_KEY` first.")

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
if "gt_lines" not in st.session_state:
    st.session_state.gt_lines = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "downloaded_adapters" not in st.session_state:
    st.session_state.downloaded_adapters = []
if "active_backend" not in st.session_state:
    st.session_state.active_backend = _active_backend

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _render_trace_step(trace, backend_type: str = "local") -> None:
    """Render a single AgenticTrace entry inside the Streamlit expander."""
    STEP_CONFIG = {
        "INIT": ("🔍", st.info, "blue"),
        "REFLECT": ("🪞", st.warning, "orange"),
        "FILTER": ("🔎", st.info, "grey"),
        "REFINE": ("✏️", st.success, "green"),
        "TERMINATE": ("🛑", st.error, "red"),
    }
    icon, box_fn, _ = STEP_CONFIG.get(trace.step_type, ("❓", st.info, "grey"))

    if backend_type == "openrouter":
        adapter_badge = "⚪ **Adapter: N/A (API)**"
    else:
        adapter_badge = (
            "🟢 **Adapter: ON**"
            if trace.adapter_state == "ON"
            else "🔴 **Adapter: OFF**"
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
    ground_truth_file,
    verbose: bool = False,
    max_new_tokens: int = 768,
    use_lexical_correction: bool = False,
) -> None:
    """
    Lazy-import and execute the full pipeline. Called only when the user
    clicks "Run Pipeline."
    """
    import logging
    import tempfile
    from io import StringIO

    from src.evaluation.evaluator import Evaluator
    from src.ingestion.pdf_handler import PDFHandler
    from src.logger.trace_logger import TraceLogger
    from src.model_engine.backend_factory import create_backend
    from src.orchestrator.agentic_orchestrator import AgenticOrchestrator
    from src.postprocessing.lexical_processor import LexicalProcessor
    from src.prompt_manager.prompt_registry import PromptRegistry

    # ── Load backend configuration from config.yaml ──────────────────────
    import yaml

    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # ── Parse ground truth text ─────────────────────────────────────────
    # Treat the entire file content as the GT for the first page
    # (1-to-1 mapping: one GT file = one document transcription)
    gt_lines: list[str] = []
    if ground_truth_file is not None:
        raw = ground_truth_file.read().decode("utf-8")
        gt_lines = [raw.strip()]

    # ── Create log window for verbose output ─────────────────────────────
    log_container = st.empty()
    log_output = StringIO()

    class StreamlitLogHandler(logging.Handler):
        """Custom logging handler that writes to Streamlit UI."""

        def emit(self, record):
            msg = self.format(record)
            log_output.write(msg + "\n")
            with log_container:
                # For OpenRouter, show API logs in a prominent expander
                expander_label = (
                    "📡 OpenRouter API Request/Response Logs"
                    if _active_backend == "openrouter"
                    else "📋 Generation Logs"
                )
                with st.expander(expander_label, expanded=True):
                    st.code(
                        log_output.getvalue() or "Waiting for logs...",
                        language="log",
                        line_numbers=True,
                    )

    # Set up handler for model_executor logger
    _loggers_to_watch = []
    if verbose:
        log_handler = StreamlitLogHandler()
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        _logger_names = (
            ["src.model_engine.openrouter_backend"]
            if _active_backend == "openrouter"
            else ["src.model_engine.model_executor"]
        )
        for _logger_name in _logger_names:
            _lg = logging.getLogger(_logger_name)
            _lg.addHandler(log_handler)
            _lg.setLevel(
                logging.DEBUG if _active_backend == "openrouter" else logging.INFO
            )
            _loggers_to_watch.append((_lg, log_handler))

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

    # ── Step 2: Load backend ───────────────────────────────────────────────
    _backend_load_text = (
        "☁️ Connecting to OpenRouter API…"
        if _active_backend == "openrouter"
        else "🤖 Loading local model weights…"
    )
    progress_bar.progress(25, text=_backend_load_text)

    try:
        backend = create_backend(config)
    except (KeyError, ValueError, EnvironmentError) as exc:
        st.error(f"Failed to initialize backend: {exc}")
        progress_bar.empty()
        return

    # Allow the UI slider to override the config.yaml max_tokens at runtime
    if _active_backend == "openrouter" and hasattr(backend, "_max_tokens"):
        backend._max_tokens = max_new_tokens

    registry = PromptRegistry(prompts_dir=str(ROOT / "prompts"))

    # ── Instantiate evaluator ────────────────────────────────────────────
    evaluator = Evaluator(output_dir=str(ROOT / "data" / "output_images"))

    # ── Instantiate lexical processor (if enabled) ───────────────────────
    lexical_processor = LexicalProcessor() if use_lexical_correction else None

    # ── Step 3: Process each page ────────────────────────────────────────
    results = []
    eval_reports = []
    n_pages = len(pages)

    for idx, page in enumerate(pages):
        progress_bar.progress(
            int(25 + 70 * (idx / n_pages)),
            text=f"🔍 Processing page {idx + 1}/{n_pages}…",
        )
        trace_logger = TraceLogger(logs_dir=str(ROOT / "logs"), page_id=page.page_id)
        orchestrator = AgenticOrchestrator(
            executor=backend,
            registry=registry,
            trace_logger=trace_logger,
            max_iterations=max_iters,
            execution_mode=execution_mode,
        )
        result = orchestrator.run(page)

        # Apply lexical correction if enabled
        if use_lexical_correction and lexical_processor:
            result.raw_text = lexical_processor.process(result.raw_text)

        # Run evaluation if GT is available for this page
        eval_report = None
        if idx < len(gt_lines):
            eval_report = evaluator.evaluate(
                page_id=page.page_id,
                ocr_text=result.raw_text,
                gt_text=gt_lines[idx],
                image_path=page.image_path,
            )
        results.append(result)
        eval_reports.append(eval_report)

    progress_bar.progress(100, text="✅ Pipeline complete!")
    progress_bar.empty()

    # Clean up log handlers
    for _lg, _lh in _loggers_to_watch:
        _lg.removeHandler(_lh)

    st.session_state.ocr_results = results
    st.session_state.eval_reports = eval_reports
    st.session_state.gt_lines = gt_lines
    st.session_state.pipeline_done = True
    st.session_state.current_page = 0
    st.rerun()


# ---------------------------------------------------------------------------
# Trigger pipeline
# ---------------------------------------------------------------------------

if run_pipeline_button and uploaded_file is not None:
    _run_pipeline(
        uploaded_file=uploaded_file,
        execution_mode=execution_mode,
        adapter_path=adapter_path_input,
        max_iters=max_iterations,
        ground_truth_file=ground_truth_file,
        verbose=verbose,
        max_new_tokens=max_new_tokens,
        use_lexical_correction=use_lexical_correction,
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

# Retrieve eval report for current page
eval_reports = st.session_state.eval_reports
eval_report = eval_reports[current_i] if current_i < len(eval_reports) else None

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
    if result.image_path and Path(result.image_path).exists():
        st.image(result.image_path, use_column_width=True)
    else:
        # Fallback: try to reconstruct path from page_id
        candidate = Path("data/output_images") / f"{result.page_id}.png"
        if candidate.exists():
            st.image(str(candidate), use_column_width=True)
        else:
            st.warning(f"Image not found for page: `{result.page_id}`")

    # Note: Heatmap disabled — requires real word bounding boxes from OCR
    # if eval_report is not None and eval_report.error_heatmap_path:
    #     heatmap_path = Path(eval_report.error_heatmap_path)
    #     if heatmap_path.exists():
    #         st.markdown("#### 🔥 Error Heatmap")
    #         st.caption("Red highlights = words predicted incorrectly vs ground truth")
    #         st.image(str(heatmap_path), use_column_width=True)

    _adapter_info = (
        "N/A (API)"
        if _active_backend == "openrouter"
        else (
            "ON"
            if execution_mode
            in (ExecutionMode.ADAPTER_ONE_SHOT, ExecutionMode.ADAPTER_REACT)
            else "OFF"
        )
    )
    st.caption(
        f"**Mode:** `{result.execution_mode.value}` | "
        f"**Backend:** `{_active_backend}` | "
        f"**Iterations:** {len([t for t in result.agentic_trace if t.step_type == 'REFINE'])} | "
        f"**Adapter:** {_adapter_info} | "
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

    # Display evaluation metrics if available
    if eval_report is not None:
        st.markdown("#### 📊 Evaluation Metrics")
        m1, m2 = st.columns(2)
        m1.metric(
            "Character Error Rate (CER)",
            f"{eval_report.cer_score:.2f}%",
            help="CER = (Substitutions + Deletions + Insertions) / Reference Length × 100. Lower is better (0% = perfect match).",
        )
        m2.metric(
            "Word Error Rate (WER)",
            f"{eval_report.wer_score:.2f}%",
            help="WER = (Substitutions + Deletions + Insertions) / Reference Word Count × 100. Lower is better (0% = perfect match).",
        )

        # Display character-level diff visualization
        if eval_report.char_diff_html:
            st.markdown("#### 🔍 Character-Level Comparison")
            st.caption(
                "Red strikethrough = missing from OCR (in ground truth) | Green = extra in OCR (hallucination)"
            )

            # CSS styling for the semantic diff display
            st.markdown(
                """
                <style>
                .semantic-diff {
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 2;
                    background: #6c757d;
                    padding: 15px;
                    border-radius: 5px;
                    border: 1px solid #dee2e6;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }
                .semantic-diff .deletion {
                    background-color: #f8d7da;
                    color: #721c24;
                    text-decoration: line-through;
                    padding: 1px 3px;
                    border-radius: 2px;
                }
                .semantic-diff .insertion {
                    background-color: #d4edda;
                    color: #155724;
                    text-decoration: none;
                    padding: 1px 3px;
                    border-radius: 2px;
                }
                .semantic-diff span {
                    color: #ffffff;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            # Display the semantic diff HTML
            st.markdown(
                f'<div class="semantic-diff">{eval_report.char_diff_html}</div>',
                unsafe_allow_html=True,
            )

        # Display character confusion matrix
        if eval_report.frequent_errors:
            st.markdown("#### 🔤 Character Substitution Patterns")
            st.caption(
                "Shows systematic errors (e.g., long-s → f in historical manuscripts)"
            )

            error_rows = []
            for gt_char, predictions in eval_report.frequent_errors.items():
                for pred_char, count in predictions.items():
                    display_gt = "·space·" if gt_char == " " else gt_char
                    display_pred = "·space·" if pred_char == " " else pred_char
                    error_rows.append(
                        f"**{display_gt}** → **{display_pred}** : {count} times"
                    )

            st.markdown("\n".join(error_rows[:15]))  # Show top 15 errors

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
            _render_trace_step(trace, backend_type=_active_backend)
