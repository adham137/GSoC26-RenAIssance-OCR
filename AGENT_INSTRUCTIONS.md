# AGENT_INSTRUCTIONS.md
# Agentic OCR Framework — Instructions for AI Coding Agents

> **Audience:** This document is written for AI coding agents (Cursor, Claude Code,
> GitHub Copilot Workspace, etc.) that will be asked to implement, extend, or refactor
> this codebase. Read this file **in full** before writing a single line of code.
> These rules are non-negotiable; violating them will cause silent correctness failures
> that are extremely hard to debug.

---

## 0. Project Summary

This is a research-grade OCR pipeline for extracting text from degraded 17th-century
Spanish manuscripts. It implements the **OCR-Agent** framework (arXiv:2602.21053v1),
which uses two reflection mechanisms inside a ReAct loop:

1. **Capability Reflection** — diagnose errors, produce a correction plan, filter
   infeasible actions.
2. **Memory Reflection** — review past iterations to avoid repeating failed strategies.

The pipeline is built around a **Qwen2.5-VL-2B-Instruct** Vision-Language Model with
optional domain-specific **LoRA adapters** for historical glyph recognition.

---

## 1. CRITICAL RULE: Dynamic LoRA Adapter Switching

This is the single most important architectural invariant in the codebase.
**Every agent must understand and preserve it.**

### The Rule

```
┌──────────────────────────────────────┬────────────────────────────────┐
│ Step / Method                        │ Required Adapter State         │
├──────────────────────────────────────┼────────────────────────────────┤
│ ModelExecutor.extract_text()         │ ON  — call set_adapter() first │
│ ModelExecutor.guided_refinement()    │ ON  — call set_adapter() first │
│ ModelExecutor.diagnose_errors()      │ OFF — call disable_adapter()   │
│ ModelExecutor.filter_plan()          │ OFF — stays disabled           │
└──────────────────────────────────────┴────────────────────────────────┘
```

### Rationale

Domain-specific LoRA weights specialise the model for historical glyph recognition
(long-s / f confusion, abbreviation expansion, faded ink). During the *planning phase*,
we explicitly want the **base model's general reasoning** so that the correction plan
is grounded in universal language understanding, not manuscript-specific biases.

### How to check compliance

Search for every call to `executor.diagnose_errors` and `executor.filter_plan` in
`src/orchestrator/agentic_orchestrator.py`. There **must** be a `executor.disable_adapter()`
call before each group, and a `executor.set_adapter()` call before every
`extract_text` / `guided_refinement` call.

### What you MUST NOT do

```python
# ❌ WRONG — adapter is not disabled before planning
result = executor.extract_text(page, prompt)
plan   = executor.diagnose_errors(page, result, reflect_prompt)  # adapter still ON!

# ✅ CORRECT
executor.set_adapter()
result = executor.extract_text(page, prompt)
executor.disable_adapter()
plan   = executor.diagnose_errors(page, result, reflect_prompt)  # adapter OFF ✓
```

---

## 2. CRITICAL RULE: No Prompt Text in Python Files

**All prompt strings must live in `prompts/*.txt` files.**

The `PromptRegistry` in `src/prompt_manager/prompt_registry.py` is the sole entry
point for prompt text. It loads, versions, and injects variables into templates.

### What you MUST NOT do

```python
# ❌ WRONG — hardcoded prompt in Python
prompt = "You are an expert paleographer. Transcribe the following image..."
result = executor.extract_text(page, prompt)

# ❌ WRONG — f-string prompt construction in Python
prompt = f"Review your transcription: {current_text}. Fix errors related to..."
```

### What you MUST do

```python
# ✅ CORRECT — load from registry with variable injection
prompt = registry.render(
    "capability_reflection",
    current_text     = current_text,
    past_reflections = "\n---\n".join(memory.past_reflections),
    iteration        = str(i),
)
```

### Prompt file format

```
# version: X.Y.Z          ← required version comment on line 1
# (other comment lines ignored)

Your prompt text here.
Use {variable_name} for runtime injection.
```

### Versioning

When you modify a prompt, bump its version number in the first line comment.
The `PromptRegistry.get_version()` method is used by the evaluation harness to
track which prompt version produced a given CER score.

---

## 3. RULE: Qwen-VL Handles Dynamic Aspect Ratios — No Manual Tiling

Qwen2.5-VL uses a native dynamic resolution mechanism. It automatically partitions
images into variable-size patches based on the input resolution.

**Do NOT write any of the following:**

```python
# ❌ WRONG — manual tiling
tiles = [image.crop((x, y, x+512, y+512)) for x, y in grid_coords]

# ❌ WRONG — forced resize before inference
image = image.resize((1024, 1024))

# ❌ WRONG — explicit chunking logic
chunks = split_image_into_chunks(image, chunk_size=512)
```

**Do this instead:**

```python
# ✅ CORRECT — pass PIL image directly to processor
from PIL import Image
image   = Image.open(page.image_path).convert("RGB")
inputs  = processor(text=[prompt], images=[image], return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024)
```

The processor handles all internal tiling. Adding external tiling logic will
**break** the model's positional embedding alignment and degrade OCR quality.

---

## 4. AgenticTrace Log Format for Streamlit UI

The Streamlit UI in `ui/app.py` renders `OCRResult.agentic_trace` using the
`_render_trace_step()` function. For correct rendering, every `AgenticTrace`
object **must** follow this schema:

```python
AgenticTrace(
    iteration       = i,            # int: 0 = init, 1+ = loop iterations
    step_type       = "REFLECT",    # str: one of INIT|REFLECT|FILTER|REFINE|TERMINATE
    adapter_state   = "OFF",        # str: "ON" or "OFF" — NEVER a boolean
    thought         = "...",        # str: the prompt or chain-of-thought (can be long)
    action          = "diagnose_errors",  # str: method name called
    observation     = "...",        # str: the model's output (will be truncated to 600 chars in UI)
    memory_snapshot = ["...", "..."]  # list[str]: copy of memory.past_reflections at this step
)
```

**step_type values and their UI rendering:**

| step_type  | Streamlit box colour | Icon | Meaning                          |
|------------|---------------------|------|----------------------------------|
| INIT       | Blue (info)         | 🔍   | Initial zero-shot extraction     |
| REFLECT    | Orange (warning)    | 🪞   | Capability Reflection / planning |
| FILTER     | Grey (info)         | 🔎   | Feasibility filtering            |
| REFINE     | Green (success)     | ✏️   | Guided Refinement                |
| TERMINATE  | Red (error)         | 🛑   | Early exit due to stagnation     |

**Do NOT:**
- Use boolean values for `adapter_state` (will crash the badge renderer).
- Add new `step_type` values without updating `STEP_CONFIG` in `ui/app.py`.
- Store the full prompt in `observation` (use `thought` for prompts).

---

## 5. Data Model Rules

All inter-module data is typed with Pydantic v2 models defined in `src/models.py`.

- **Never** pass raw dicts between modules where a typed model exists.
- `DocumentPage.image_path` stores a **file path string**, not raw binary.
- `AgenticMemory` is **mutable** — call `record_iteration()` after each loop step.
- `OCRResult.agentic_trace` is the **single source of truth** for the UI and for
  offline analysis — ensure it is always populated before returning.

---

## 6. Module Boundary Rules

```
PDFHandler  →  DocumentPage  →  ModelExecutor  →  AgenticOrchestrator
                                                         ↓
                                                     OCRResult
                                                         ↓
                                                      Evaluator  →  EvaluationReport
```

- `PDFHandler` must **never** import from `model_engine` or `orchestrator`.
- `ModelExecutor` must **never** directly read prompt files — it receives rendered
  prompt strings as arguments.
- `AgenticOrchestrator` is the **only** module allowed to call both `set_adapter()`
  and `disable_adapter()`. No other module should touch adapter state.
- `Evaluator` is **read-only** with respect to model state — it never calls the VLM.

---

## 7. Implementing a New Execution Mode

To add a new mode (e.g., `MULTI_AGENT_REACT`):

1. Add the enum value to `ExecutionMode` in `src/models.py`.
2. Add a new `elif` branch in `AgenticOrchestrator.run()`.
3. Implement the corresponding `_run_<mode>()` private method.
4. Update the Streamlit sidebar `selectbox` options in `ui/app.py`.
5. Write a corresponding test in `tests/test_pipeline.py`.
6. **Do not** modify existing mode implementations when adding new ones.

---

## 8. Adding a New Prompt Template

1. Create `prompts/<new_key>.txt` with `# version: 1.0.0` on line 1.
2. Use `{variable_name}` placeholders for runtime injection.
3. Call `registry.render("<new_key>", variable_name="value")` in the orchestrator.
4. **Never** add the prompt text to any Python file.
5. Document the expected variables in a comment block at the top of the `.txt` file.

---

## 9. Test Writing Conventions

- All tests live in `tests/test_pipeline.py` (or new files for new modules).
- Mock the VLM entirely — tests must pass on CPU without model weights.
- Use `pytest.importorskip("peft")` to gracefully skip GPU-only tests in CI.
- Use the `MockModelExecutor` pattern from `test_dynamic_lora_switching` for any
  test that needs to assert adapter state at inference time.
- Tests that require `PDFHandler` implementation should be marked `pytest.skip()`
  with a clear message until the implementation is complete.

---

## 10. Logging Conventions

- All modules use `logging.getLogger(__name__)` — never `print()`.
- The `TraceLogger` in `src/logger/trace_logger.py` handles structured agentic logs.
- Log levels:
  - `DEBUG` — per-step adapter state, token counts, cache hits
  - `INFO`  — page-level events (loaded, processed, saved)
  - `WARNING` — recoverable issues (missing GT, image validation failures)
  - `ERROR` — pipeline failures (model load error, file not found)

---

## 11. File Layout Reference

```
agentic_ocr_framework/
├── src/
│   ├── models.py                         ← ALL data types (edit with care)
│   ├── ingestion/
│   │   └── pdf_handler.py                ← PDFHandler
│   ├── prompt_manager/
│   │   └── prompt_registry.py            ← PromptRegistry (NO PROMPT TEXT HERE)
│   ├── model_engine/
│   │   └── model_executor.py             ← ModelExecutor + adapter management
│   ├── orchestrator/
│   │   └── agentic_orchestrator.py       ← ReAct loop (adapter switching lives here)
│   ├── evaluation/
│   │   └── evaluator.py                  ← CER/WER/heatmap
│   └── logger/
│       └── trace_logger.py               ← JSONL structured logging
├── tests/
│   └── test_pipeline.py                  ← All pytest tests
├── ui/
│   └── app.py                            ← Streamlit GUI
├── prompts/                              ← ALL prompt text lives here
│   ├── initial_extraction.txt
│   ├── capability_reflection.txt
│   ├── capability_filter.txt
│   ├── memory_reflection.txt
│   └── guided_refinement.txt
├── data/
│   ├── raw_pdfs/                         ← Input documents
│   ├── ground_truth/                     ← GT .txt files (one per PDF)
│   └── output_images/                    ← Rasterised pages + heatmaps
├── logs/                                 ← JSONL trace files (auto-generated)
├── configs/                              ← YAML experiment configs (future)
├── notebooks/                            ← Analysis notebooks (future)
├── requirements.txt
├── README.md
└── AGENT_INSTRUCTIONS.md                 ← This file
```
