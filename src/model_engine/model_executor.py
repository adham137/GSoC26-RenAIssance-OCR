"""
src/model_engine/model_executor.py
===================================
Model Execution & Adapter Management Module (System Design §2.3).

This is the core inference engine. It is responsible for:
    1. Loading the Qwen-VL base model via Hugging Face ``transformers``.
    2. Dynamically attaching / detaching LoRA adapters via ``peft``.
    3. Executing inference for a given image + prompt pair.
    4. Exposing a clean API that the :class:`~src.orchestrator.AgenticOrchestrator`
       can call without knowing *which* state the adapter is in.

=============================================================================
CRITICAL ARCHITECTURAL RULE — DYNAMIC LoRA SWITCHING
=============================================================================
The adapter state is **not** fixed for an entire pipeline run.  It switches
*within* a single agentic iteration based on the cognitive task:

    ┌──────────────────────────────┬────────────────┐
    │ Operation                    │ Adapter State  │
    ├──────────────────────────────┼────────────────┤
    │ extract_text (initial read)  │  ON  ✓         │
    │ diagnose_errors (Capability  │  OFF ✗         │
    │   Reflection / planning)     │                │
    │ filter_plan                  │  OFF ✗         │
    │ guided_refinement (Refine)   │  ON  ✓         │
    └──────────────────────────────┴────────────────┘

Rationale: domain-specific LoRA weights specialise the model for
historical glyph recognition.  During the *planning/reflection phase*,
we want the model's general reasoning capability unbiased by those
specialisations so it produces broadly feasible correction plans.

Any future developer adding new step types MUST explicitly declare
which adapter state is required.  See ``set_adapter()`` / ``disable_adapter()``.
=============================================================================

Implementation Notes
--------------------
* Model: ``Qwen/Qwen2.5-VL-2B-Instruct`` (configurable).
* Qwen-VL handles dynamic aspect ratios natively — do **not** write
  manual image chunking / tiling logic.  Pass the PIL image directly.
* Load in 4-bit via ``bitsandbytes`` BnB config for 24 GB VRAM budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.models import DocumentPage

logger = logging.getLogger(__name__)


class ModelExecutor:
    """
    Thin wrapper around a Qwen-VL model that manages both inference
    and LoRA adapter lifecycle.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
        Default: ``"Qwen/Qwen2.5-VL-2B-Instruct"``.
    adapter_path : str | Path | None
        Path to the directory containing LoRA adapter weights
        (``adapter_config.json`` + ``adapter_model.bin``).
        If ``None``, the model runs without any adapter.
    load_in_4bit : bool
        Whether to quantise to 4-bit using ``bitsandbytes``.
        Recommended for consumer GPUs.
    device : str
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.

    Attributes
    ----------
    _model : Any
        The loaded ``transformers`` / ``peft`` model object.
    _processor : Any
        The Qwen-VL processor (tokeniser + image processor).
    _adapter_loaded : bool
        Whether a LoRA adapter is currently active.

    Examples
    --------
    >>> executor = ModelExecutor(adapter_path="models/lora_v2")
    >>> executor.set_adapter()
    >>> text = executor.extract_text(page, "Transcribe all visible text.")
    >>> executor.disable_adapter()
    >>> plan = executor.diagnose_errors(page, text, memory)
    """

    DEFAULT_MODEL = "models/Qwen3.5-0.8B"

    # Maximum image dimension to prevent excessive visual tokens
    MAX_IMAGE_DIM = 1024

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 2048,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.verbose = verbose

        # Set by _load_model()
        self._model: Any = None
        self._processor: Any = None
        self._adapter_loaded: bool = False
        self.max_new_tokens = max_new_tokens

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Initialise the VLM and (optionally) pre-load the LoRA adapter.

        Must be called once before any inference method is invoked.

        Raises
        ------
        RuntimeError
            If the model or adapter weights cannot be loaded.
        """
        logger.info(
            "Loading model '%s'  (4-bit=%s, adapter=%s)",
            self.model_name,
            self.load_in_4bit,
            str(self.adapter_path) if self.adapter_path else "None",
        )

        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch bitsandbytes accelerate"
            ) from exc

        # ── Processor ────────────────────────────────────────────────────
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        logger.debug("Processor loaded.")

        # ── BitsAndBytes 4-bit config ─────────────────────────────────────
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # ── Base model ────────────────────────────────────────────────────
        base_model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.load_in_4bit else None,
        )
        logger.debug("Base model loaded.")

        # ── LoRA adapter ──────────────────────────────────────────────────
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "peft is required for LoRA support. Install with: pip install peft"
                ) from exc

            if not self.adapter_path.exists():
                raise RuntimeError(
                    f"Adapter path does not exist: {self.adapter_path}\n"
                    "Ensure adapter_config.json and weights are present."
                )
            self._model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            self._adapter_loaded = True
            logger.info("LoRA adapter attached from '%s'", self.adapter_path)
        else:
            self._model = base_model
            self._adapter_loaded = False
            logger.info("Running as base model (no adapter).")

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def set_adapter(self) -> None:
        """
        Activate the LoRA adapter weights.

        Called before tasks that benefit from domain-specific glyph
        knowledge: ``extract_text`` and ``guided_refinement``.

        Raises
        ------
        RuntimeError
            If no ``adapter_path`` was provided at construction time.
        """
        if self._model is None:
            raise RuntimeError("load_model() must be called before set_adapter().")
        if self.adapter_path is None:
            raise RuntimeError(
                "Cannot enable adapter: no adapter_path was provided at construction."
            )
        self._model.enable_adapter_layers()
        self._adapter_loaded = True
        logger.debug("LoRA adapter ENABLED ✓")

    def disable_adapter(self) -> None:
        """
        Deactivate the LoRA adapter, reverting to pure base-model weights.

        Called before tasks that require general reasoning:
        ``diagnose_errors`` and ``filter_plan``.
        """
        if self._model is None:
            raise RuntimeError("load_model() must be called before disable_adapter().")
        if self.adapter_path is not None and self._adapter_loaded:
            self._model.disable_adapter_layers()
            self._adapter_loaded = False
            logger.debug("LoRA adapter DISABLED ✗")

    @property
    def adapter_state(self) -> str:
        """Return ``"ON"`` if the adapter is currently active, else ``"OFF"``."""
        return "ON" if self._adapter_loaded else "OFF"

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    def extract_text(
        self,
        page: DocumentPage,
        prompt: str,
    ) -> str:
        """
        Perform a single OCR pass on ``page`` using the provided ``prompt``.

        Adapter should be **ON** when this is called.
        """
        self._assert_loaded("extract_text")
        logger.debug("extract_text called (adapter=%s)", self.adapter_state)
        return self._run_vision_inference(page.image_path, prompt)

    def diagnose_errors(
        self,
        page: DocumentPage,
        current_text: str,
        reflection_prompt: str,
    ) -> str:
        """
        Ask the model to identify transcription errors and produce a
        correction plan (Capability Reflection, §3.2).

        Adapter should be **OFF** when this is called.
        """
        self._assert_loaded("diagnose_errors")
        logger.debug("diagnose_errors called (adapter=%s)", self.adapter_state)
        return self._run_vision_inference(page.image_path, reflection_prompt)

    def filter_plan(
        self,
        raw_plan: str,
        filter_prompt: str,
    ) -> str:
        """
        Apply the capability-filter to remove infeasible actions from
        the raw correction plan (φ(a) filter from §3.2).

        Adapter should be **OFF** when this is called.
        Text-only — no image needed for filtering.
        """
        self._assert_loaded("filter_plan")
        logger.debug("filter_plan called (adapter=%s)", self.adapter_state)
        return self._run_text_inference(filter_prompt)

    def guided_refinement(
        self,
        page: DocumentPage,
        feasible_plan: str,
        refinement_prompt: str,
    ) -> str:
        """
        Execute the feasible correction plan to produce a refined
        transcription (Guided Refinement, §3.3).

        Adapter should be **ON** when this is called.
        """
        self._assert_loaded("guided_refinement")
        logger.debug("guided_refinement called (adapter=%s)", self.adapter_state)
        return self._run_vision_inference(page.image_path, refinement_prompt)

    # ------------------------------------------------------------------
    # Internal inference helpers
    # ------------------------------------------------------------------

    def _resize_image_if_needed(self, image: "Image.Image") -> "Image.Image":
        """
        Resize image if it exceeds MAX_IMAGE_DIM to prevent excessive visual tokens.

        Qwen3.5-VL's vision encoder creates ~4 tokens per 14x14 patch.
        A 2657x3728 image would create ~38k tokens which causes OOM/hanging.
        We cap the largest dimension to MAX_IMAGE_DIM (1024) to keep tokens manageable.
        """
        from PIL import Image

        width, height = image.size
        max_dim = max(width, height)

        if max_dim > self.MAX_IMAGE_DIM:
            scale = self.MAX_IMAGE_DIM / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Ensure dimensions are divisible by 14 (ViT patch size)
            new_width = (new_width // 14) * 14
            new_height = (new_height // 14) * 14
            logger.info(
                "Resizing image from %dx%d to %dx%d to prevent excessive tokens",
                width,
                height,
                new_width,
                new_height,
            )
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def _run_vision_inference(self, image_path: str, prompt: str, verbose: bool | None = None) -> str:
        """
        Run a vision+text inference pass using the Qwen-VL chat template.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        prompt : str
            Text prompt for the model.
        verbose : bool, optional
            If True, log generation progress including token count and timing.
            If None, uses self.verbose.

        Returns
        -------
        str
            Generated text response.

        Optimizations
        -------------
        * Uses early stopping via eos_token_id to prevent runaway generation
        * Verbose mode logs token generation stats for debugging
        """
        import time

        import torch
        from PIL import Image

        # Use instance verbose if not overridden
        if verbose is None:
            verbose = self.verbose

        # Load and potentially resize image
        image = Image.open(image_path).convert("RGB")
        image = self._resize_image_if_needed(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # process_vision_info extracts image tensors from the message dict
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            # Fallback: pass image directly when qwen_vl_utils is unavailable
            image_inputs, video_inputs = [image], None

        processor_kwargs: dict = dict(
            text=[text_input],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        if video_inputs is not None:
            processor_kwargs["videos"] = video_inputs

        inputs = self._processor(**processor_kwargs).to(self._model.device)

        if verbose:
            logger.info(
                "Starting generation: input_tokens=%d, max_new_tokens=%d",
                inputs.input_ids.shape[1],
                self.max_new_tokens,
            )

        start_time = time.time()

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                do_sample=False,  # greedy — deterministic
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_scores=False,
            )

        elapsed = time.time() - start_time
        generated = output_ids.sequences[:, inputs.input_ids.shape[1] :]
        num_tokens = generated.shape[1]

        if verbose:
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                "Generation complete: %d tokens in %.2fs (%.2f tokens/sec)",
                num_tokens,
                elapsed,
                tokens_per_sec,
            )

        decoded = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded[0].strip()

    def _run_text_inference(self, prompt: str, verbose: bool | None = None) -> str:
        """
        Run a text-only inference pass (no image).

        Used for ``filter_plan`` where we only need the model to filter
        a text plan — no visual input required.
        """
        import time

        import torch

        # Use instance verbose if not overridden
        if verbose is None:
            verbose = self.verbose

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        if verbose:
            logger.info(
                "Starting text generation: input_tokens=%d, max_new_tokens=512",
                inputs.input_ids.shape[1],
            )

        start_time = time.time()

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=512,  # plans are short
                eos_token_id=self._processor.tokenizer.eos_token_id,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
            )

        elapsed = time.time() - start_time
        generated = output_ids.sequences[:, inputs.input_ids.shape[1] :]
        num_tokens = generated.shape[1]

        if verbose:
            tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
            logger.info(
                "Text generation complete: %d tokens in %.2fs (%.2f tokens/sec)",
                num_tokens,
                elapsed,
                tokens_per_sec,
            )

        decoded = self._processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded[0].strip()

    def _assert_loaded(self, method_name: str) -> None:
        """Raise RuntimeError if the model has not been initialised yet."""
        if self._model is None or self._processor is None:
            raise RuntimeError(
                f"'{method_name}()' was called before load_model(). "
                "Call executor.load_model() once during startup."
            )
