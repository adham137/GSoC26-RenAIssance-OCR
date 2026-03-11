"""
src/evaluation/evaluator.py
============================
Evaluation & Error Analysis Module (System Design §2.5).

Responsibilities
----------------
* Compute Character Error Rate (CER) and Word Error Rate (WER) between
  OCR output and a Ground Truth (GT) string.
* Build a character-level confusion matrix from Levenshtein edit operations.
* Generate a visual error-overlay image highlighting mis-transcribed words
  on the original manuscript scan.

Metric Definitions
------------------
Both CER and WER are defined as normalised edit distance:

    CER = (S_c + D_c + I_c) / N_c
    WER = (S_w + D_w + I_w) / N_w

where S = substitutions, D = deletions, I = insertions, N = reference length.

Dependencies
------------
``jiwer`` is the canonical library for WER/CER calculation.  It handles
normalisation (lowercasing, punctuation stripping) via its built-in
``Compose`` transforms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.models import EvaluationReport

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Computes OCR quality metrics and generates visual error reports.

    Parameters
    ----------
    output_dir : str | Path
        Directory where error-overlay images are saved.
        Defaults to ``data/output_images/eval/``.
    normalise : bool
        If ``True``, apply standard text normalisation before scoring
        (lowercase, strip punctuation, collapse whitespace).

    Examples
    --------
    >>> evaluator = Evaluator()
    >>> report = evaluator.evaluate(
    ...     page_id    = "manuscript_1640_page_001",
    ...     ocr_text   = "Thi5 is a t3st.",
    ...     gt_text    = "This is a test.",
    ...     image_path = "data/output_images/manuscript_1640_page_001.png",
    ... )
    >>> print(f"CER: {report.cer_score:.4f}  WER: {report.wer_score:.4f}")
    CER: 0.1333  WER: 0.5000
    """

    def __init__(
        self,
        output_dir : str | Path = "data/output_images/eval",
        normalise  : bool       = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.normalise  = normalise
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        page_id    : str,
        ocr_text   : str,
        gt_text    : str,
        image_path : str | Path | None = None,
    ) -> EvaluationReport:
        """
        Full evaluation pass: metrics + confusion matrix + (optional) heatmap.

        Parameters
        ----------
        page_id : str
            Identifier linking the report back to a :class:`~src.models.DocumentPage`.
        ocr_text : str
            The OCR system's output transcription.
        gt_text : str
            The human-verified ground truth string.
        image_path : str | Path | None
            Path to the original manuscript image.  If provided, an
            error-overlay visualisation is generated.

        Returns
        -------
        EvaluationReport
            Populated report with all metrics and an optional heatmap path.
        """
        if self.normalise:
            ocr_text = self._normalise_text(ocr_text)
            gt_text  = self._normalise_text(gt_text)

        cer = self.compute_cer(ocr_text, gt_text)
        wer = self.compute_wer(ocr_text, gt_text)

        logger.info("Page '%s' — CER=%.4f  WER=%.4f", page_id, cer, wer)

        frequent_errors = self.build_confusion_matrix(ocr_text, gt_text)

        heatmap_path: str | None = None
        if image_path is not None:
            heatmap_path = self.generate_error_heatmap(
                page_id    = page_id,
                ocr_text   = ocr_text,
                gt_text    = gt_text,
                image_path = Path(image_path),
            )

        return EvaluationReport(
            page_id            = page_id,
            cer_score          = cer,
            wer_score          = wer,
            frequent_errors    = frequent_errors,
            error_heatmap_path = heatmap_path,
        )

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def compute_cer(self, hypothesis: str, reference: str) -> float:
        """
        Compute Character Error Rate (CER).

        Parameters
        ----------
        hypothesis : str
            OCR output (prediction).
        reference : str
            Ground truth string.

        Returns
        -------
        float
            CER in range [0, ∞).  Values > 1 are possible when the
            hypothesis is longer than the reference.

        """
        try:
            from jiwer import cer as jiwer_cer
        except ImportError as exc:
            raise ImportError(
                "jiwer is required for CER computation. "
                "Install with: pip install jiwer"
            ) from exc

        if not reference:
            logger.warning("Empty reference string — CER is undefined, returning 0.0")
            return 0.0

        return float(jiwer_cer(reference, hypothesis))

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        """
        Compute Word Error Rate (WER).

        Parameters
        ----------
        hypothesis : str
            OCR output (prediction).
        reference : str
            Ground truth string.

        Returns
        -------
        float
            WER in range [0, ∞).

        """
        try:
            from jiwer import wer as jiwer_wer
        except ImportError as exc:
            raise ImportError(
                "jiwer is required for WER computation. "
                "Install with: pip install jiwer"
            ) from exc

        if not reference:
            logger.warning("Empty reference string — WER is undefined, returning 0.0")
            return 0.0

        return float(jiwer_wer(reference, hypothesis))

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def build_confusion_matrix(
        self,
        hypothesis : str,
        reference  : str,
    ) -> dict[str, dict[str, int]]:
        """
        Build a character-level confusion matrix from Levenshtein edit ops.

        Parameters
        ----------
        hypothesis : str
            OCR output.
        reference : str
            Ground truth.

        Returns
        -------
        dict[str, dict[str, int]]
            Nested mapping ``{gt_char: {predicted_char: count}}``.
            Example: ``{"s": {"f": 3}}`` means 17th-century long-s was
            mistaken for 'f' three times.

        Implementation sketch
        ---------------------
        Use ``Levenshtein.editops(hypothesis, reference)`` to get the
        raw operation list, then tally substitution pairs.

        ```
        """
        try:
            import Levenshtein
        except ImportError as exc:
            raise ImportError(
                "python-Levenshtein is required for confusion matrix. "
                "Install with: pip install python-Levenshtein"
            ) from exc

        matrix: dict[str, dict[str, int]] = {}

        # editops returns a list of (op_type, hyp_pos, ref_pos) tuples
        # op_type is one of: 'insert', 'delete', 'replace'
        for op, hyp_pos, ref_pos in Levenshtein.editops(hypothesis, reference):
            if op == "replace":
                gt_char   = reference[ref_pos]
                pred_char = hypothesis[hyp_pos]
                if gt_char not in matrix:
                    matrix[gt_char] = {}
                matrix[gt_char][pred_char] = matrix[gt_char].get(pred_char, 0) + 1

        # Sort each inner dict by count descending for readability
        return {
            gt_char: dict(sorted(preds.items(), key=lambda x: x[1], reverse=True))
            for gt_char, preds in sorted(matrix.items())
        }

    def generate_error_heatmap(
        self,
        page_id    : str,
        ocr_text   : str,
        gt_text    : str,
        image_path : Path,
    ) -> str:
        """
        Generate a visual error overlay on the manuscript image.

        Words that differ between ``ocr_text`` and ``gt_text`` are
        highlighted with semi-transparent red bounding boxes.

        Parameters
        ----------
        page_id : str
            Used to construct the output filename.
        ocr_text : str
            OCR output text.
        gt_text : str
            Ground truth text.
        image_path : Path
            Path to the source manuscript image.

        Returns
        -------
        str
            Absolute path to the saved heatmap PNG.

        Notes
        -----
        Word-level alignment is performed with a simple diff (e.g.
        ``difflib.SequenceMatcher``).  Character-level bounding boxes
        require a layout-aware OCR engine that returns word coordinates;
        this is a future enhancement (marked TODO).

        Implementation sketch (word-level fallback)
        -------------------------------------------
        ```
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for heatmap generation. "
                "Install with: pip install Pillow"
            ) from exc

        import difflib

        img  = Image.open(image_path).convert("RGBA")
        w, h = img.size

        # Overlay layer for semi-transparent highlights
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        ocr_words = ocr_text.split()
        gt_words  = gt_text.split()

        if not gt_words:
            logger.warning("Empty GT for heatmap — returning plain image copy.")
            out_path = self.output_dir / f"{page_id}_heatmap.png"
            img.convert("RGB").save(str(out_path))
            return str(out_path)

        # Find the word-level error positions via SequenceMatcher
        sm       = difflib.SequenceMatcher(None, ocr_words, gt_words, autojunk=False)
        opcodes  = sm.get_opcodes()

        # Collect indices of OCR words that differ from GT
        error_word_indices: set[int] = set()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                for idx in range(i1, i2):
                    error_word_indices.add(idx)

        if not error_word_indices:
            logger.info("No word-level errors found for page '%s' — heatmap is clean.", page_id)
        else:
            # Approximate word positions by distributing evenly across page width.
            # Real bounding-box support requires a layout-aware OCR pass (future work).
            n_words    = len(ocr_words)
            line_h     = max(20, h // max(n_words // 10, 1))
            word_w     = max(60, w // 20)
            words_per_row = max(1, w // (word_w + 8))

            for err_idx in error_word_indices:
                row = err_idx // words_per_row
                col = err_idx %  words_per_row
                x0  = col * (word_w + 8)
                y0  = row * (line_h + 4)
                x1  = min(x0 + word_w, w - 1)
                y1  = min(y0 + line_h, h - 1)
                # Semi-transparent red highlight
                draw.rectangle([x0, y0, x1, y1], fill=(220, 30, 30, 100))

            logger.info(
                "Heatmap for '%s': %d/%d words flagged as errors.",
                page_id, len(error_word_indices), len(ocr_words),
            )

        # Composite and save
        combined = Image.alpha_composite(img, overlay).convert("RGB")
        out_path = self.output_dir / f"{page_id}_heatmap.png"
        combined.save(str(out_path))
        logger.debug("Saved error heatmap → %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_text(text: str) -> str:
        """
        Apply standard OCR normalisation before metric calculation.

        Steps applied:
            1. Lowercase.
            2. Strip leading/trailing whitespace.
            3. Collapse internal whitespace runs to a single space.

        Parameters
        ----------
        text : str
            Raw text to normalise.

        Returns
        -------
        str
            Normalised text.
        """
        import re
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text
