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
* Generate semantic character/word-level HTML diffs for human readability.

Metric Definitions
------------------
Both CER and WER are defined as normalised edit distance:

    CER = (S_c + D_c + I_c) / N_c
    WER = (S_w + D_w + I_w) / N_w

where S = substitutions, D = deletions, I = insertions, N = reference length.

Dependencies
------------
``jiwer`` is the canonical library for WER/CER calculation.
``diff-match-patch`` provides semantic diff generation with human-readable cleanup.
"""

from __future__ import annotations

import logging
import re
import string
from pathlib import Path
from typing import TYPE_CHECKING
import unicodedata
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

    _LIGATURES = {
        "ﬁ": "fi",  # U+FB01
        "ﬂ": "fl",  # U+FB02
        "ﬃ": "ffi",  # U+FB03
        "ﬄ": "ffl",  # U+FB04
        "ﬀ": "ff",  # U+FB00
        "ﬅ": "st",  # U+FB05
        "ﬆ": "st",  # U+FB06
    }

    def __init__(
        self,
        output_dir: str | Path = "data/output_images/eval",
        normalise: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.normalise = normalise
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        page_id: str,
        ocr_text: str,
        gt_text: str,
        image_path: str | Path | None = None,
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
            gt_text = self._normalise_text(gt_text)

        cer = self.compute_cer(ocr_text, gt_text)
        wer = self.compute_wer(ocr_text, gt_text)

        logger.info("Page '%s' — CER=%.4f  WER=%.4f", page_id, cer, wer)

        frequent_errors = self.build_confusion_matrix(ocr_text, gt_text)

        # Generate semantic character-level diff HTML for visualization
        char_diff_html = self.generate_semantic_diff_html(ocr_text, gt_text)

        # Heatmap generation disabled — requires actual word bounding boxes
        # from a layout-aware OCR engine. The current grid approximation is
        # misleading as it doesn't match actual text positions.
        # TODO: Integrate EasyOCR/Tesseract for real bounding box support.
        heatmap_path: str | None = None

        return EvaluationReport(
            page_id=page_id,
            cer_score=cer,
            wer_score=wer,
            frequent_errors=frequent_errors,
            error_heatmap_path=heatmap_path,
            char_diff_html=char_diff_html,
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
            CER as a percentage [0, ∞). Values > 100 are possible when the
            hypothesis is much longer than the reference.

        """
        try:
            from jiwer import cer as jiwer_cer
        except ImportError as exc:
            raise ImportError(
                "jiwer is required for CER computation. Install with: pip install jiwer"
            ) from exc

        # Normalize both texts before comparison
        ref_norm = self._normalise_text(reference)
        hyp_norm = self._normalise_text(hypothesis)

        # Handle edge cases
        if not ref_norm and not hyp_norm:
            logger.debug("Both reference and hypothesis are empty — CER = 0.0")
            return 0.0
        if not ref_norm:
            logger.warning("Empty reference string — CER = 100.0 (100% error)")
            return 100.0

        # Multiply by 100 to return percentage (standard CER reporting)
        return float(jiwer_cer(ref_norm, hyp_norm)) * 100.0

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
            WER as a percentage [0, ∞).

        """
        try:
            from jiwer import wer as jiwer_wer
        except ImportError as exc:
            raise ImportError(
                "jiwer is required for WER computation. Install with: pip install jiwer"
            ) from exc

        # Normalize both texts before comparison
        ref_norm = self._normalise_text(reference)
        hyp_norm = self._normalise_text(hypothesis)

        # Handle edge cases
        if not ref_norm and not hyp_norm:
            logger.debug("Both reference and hypothesis are empty — WER = 0.0")
            return 0.0
        if not ref_norm:
            logger.warning("Empty reference string — WER = 100.0 (100% error)")
            return 100.0

        # Multiply by 100 to return percentage (standard WER reporting)
        return float(jiwer_wer(ref_norm, hyp_norm)) * 100.0

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def build_confusion_matrix(
        self,
        hypothesis: str,
        reference: str,
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
                gt_char = reference[ref_pos]
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
        page_id: str,
        ocr_text: str,
        gt_text: str,
        image_path: Path,
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

        img = Image.open(image_path).convert("RGBA")
        w, h = img.size

        # Overlay layer for semi-transparent highlights
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        ocr_words = ocr_text.split()
        gt_words = gt_text.split()

        if not gt_words:
            logger.warning("Empty GT for heatmap — returning plain image copy.")
            out_path = self.output_dir / f"{page_id}_heatmap.png"
            img.convert("RGB").save(str(out_path))
            return str(out_path)

        # Find the word-level error positions via SequenceMatcher
        sm = difflib.SequenceMatcher(None, ocr_words, gt_words, autojunk=False)
        opcodes = sm.get_opcodes()

        # Collect indices of OCR words that differ from GT
        error_word_indices: set[int] = set()
        for tag, i1, i2, j1, j2 in opcodes:
            if tag != "equal":
                for idx in range(i1, i2):
                    error_word_indices.add(idx)

        if not error_word_indices:
            logger.info(
                "No word-level errors found for page '%s' — heatmap is clean.", page_id
            )
        else:
            # Approximate word positions by distributing evenly across page width.
            # Real bounding-box support requires a layout-aware OCR pass (future work).
            n_words = len(ocr_words)
            line_h = max(20, h // max(n_words // 10, 1))
            word_w = max(60, w // 20)
            words_per_row = max(1, w // (word_w + 8))

            for err_idx in error_word_indices:
                row = err_idx // words_per_row
                col = err_idx % words_per_row
                x0 = col * (word_w + 8)
                y0 = row * (line_h + 4)
                x1 = min(x0 + word_w, w - 1)
                y1 = min(y0 + line_h, h - 1)
                # Semi-transparent red highlight
                draw.rectangle([x0, y0, x1, y1], fill=(220, 30, 30, 100))

            logger.info(
                "Heatmap for '%s': %d/%d words flagged as errors.",
                page_id,
                len(error_word_indices),
                len(ocr_words),
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
        Normalizes text for accurate OCR evaluation, matching standard CER calculators.
        Steps applied:
            0. Expand Unicode ligatures (ﬁ → fi, ﬂ → fl, etc.) before any other step.
            1. Strip superscript notation (^o, ^a, ^e, etc.) for evaluation purposes.
            2. Convert to lowercase.
            3. Strip combining accent marks, preserving ñ.
            Mirrors training normalization Rule 5 exactly so that the model's
            learned (accent-free) output is fairly compared against ground truth.
            4. Remove all punctuation, including Unicode punctuation (e.g. … U+2026)
            not covered by the original ASCII-only string.punctuation approach.
            5. Collapse all whitespace (newlines, tabs, multiple spaces) into a single space.
            6. Strip leading and trailing whitespace.

        Parameters
        ----------
        text : str
            Raw text to normalise.

        Returns
        -------
        str
            Normalised text.

        Examples
        --------
        >>> _normalise_text("conﬁrmar")
        'confirmar'
        >>> _normalise_text("off^o")
        'off'
        >>> _normalise_text("Milán")
        'milan'
        >>> _normalise_text("España")
        'espana'
        >>> _normalise_text("mañana")
        'mañana'
        >>> _normalise_text("…impunitatem")
        'impunitatem'
        >>> _normalise_text("Hello\\nWorld")
        'hello world'
        """
        if not text:
            return ""

        # Step 0 — expand Unicode ligatures before any character-level processing
        for ligature, expansion in Evaluator._LIGATURES.items():
            text = text.replace(ligature, expansion)

        # PP-3: Strip superscript notation for evaluation purposes
        # This removes ^ followed by 1-2 letters (including accented vowels)
        # so that "off^o" evaluates the same as "off"
        text = re.sub(r"\^[a-zA-Záéíóú]{1,2}", "", text)

        # Step 1 — lowercase
        text = text.lower()

        # Step 2 — strip combining accent marks, preserve ñ
        # Mirrors training normalization Rule 5 exactly.
        # After lowercasing we only encounter lowercase 'n', so the uppercase
        # branch from the training function is intentionally omitted.
        nfd = unicodedata.normalize("NFD", text)
        out, i = [], 0
        while i < len(nfd):
            c = nfd[i]
            if (
                c == "n" and i + 1 < len(nfd) and nfd[i + 1] == "\u0303"
            ):  # combining tilde → ñ
                out.append(c)
                out.append("\u0303")
                i += 2
            elif unicodedata.category(c) == "Mn":  # any other combining mark
                i += 1  # drop it
            else:
                out.append(c)
                i += 1
        text = unicodedata.normalize("NFC", "".join(out))

        # Step 3 — remove all punctuation, Unicode-aware
        # Replaces the original str.translate(string.punctuation) which only
        # covered 32 ASCII punctuation characters and missed e.g. … (U+2026).
        # [^\w\s] removes every character that is not a Unicode word character
        # or whitespace. ñ survives because \w with re.UNICODE matches it.
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

        # Step 4 — collapse whitespace
        normalized = re.sub(r"\s+", " ", text)
        return normalized.strip()

    def generate_semantic_diff_html(self, hypothesis: str, reference: str) -> str:
        """
        Generates an HTML diff highlighting insertions and deletions at the character/word level.

        Uses diff_match_patch with semantic cleanup for human readability.
        The diff shows:
        - Deletions (missing in hypothesis): red background with strikethrough
        - Insertions (extra in hypothesis): green background
        - Equal text: normal display

        Parameters
        ----------
        hypothesis : str
            OCR output.
        reference : str
            Ground truth.

        Returns
        -------
        str
            HTML string with highlighted differences.
        """
        try:
            from diff_match_patch import diff_match_patch
        except ImportError as exc:
            raise ImportError(
                "diff-match-patch is required for semantic diff generation. "
                "Install with: pip install diff-match-patch"
            ) from exc

        # Normalize both texts for fair comparison
        ref_norm = self._normalise_text(reference)
        hyp_norm = self._normalise_text(hypothesis)

        dmp = diff_match_patch()

        # Disable timeout to handle noisy OCR text without truncation
        dmp.Diff_Timeout = 0.0

        # Generate the raw diff
        diffs = dmp.diff_main(ref_norm, hyp_norm)

        # Note: Semantic cleanup removed to preserve exact character-level matches
        # dmp.diff_cleanupSemantic(diffs)

        # Convert diffs to HTML
        html_output = []
        for op, data in diffs:
            # Escape HTML entities in the text to prevent rendering issues
            text = data.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

            if op == dmp.DIFF_INSERT:
                # Highlight insertions (hypothesis - extra text) in green
                html_output.append(
                    f'<span style="background-color: #d4edda; color: #155724; text-decoration: none;">{text}</span>'
                )
            elif op == dmp.DIFF_DELETE:
                # Highlight deletions (reference - missing text) in red with strikethrough
                html_output.append(
                    f'<span style="background-color: #f8d7da; color: #721c24; text-decoration: line-through;">{text}</span>'
                )
            elif op == dmp.DIFF_EQUAL:
                # Leave matching text as normal
                html_output.append(f"<span>{text}</span>")

        return "".join(html_output)
