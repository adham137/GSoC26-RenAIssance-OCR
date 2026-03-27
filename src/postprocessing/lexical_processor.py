"""
Lexical Post-Processing for OCR Output.

Applies word segmentation and archaic Spanish spacing corrections to reduce
Word Error Rate (WER) in historical document transcription.
"""

import re
import unicodedata

try:
    from symspellpy import SymSpell

    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False


# =============================================================================
# PP-1: Non-Latin Character Scrubbing
# =============================================================================


def _unicode_block(char: str) -> str:
    """
    Determine the Unicode block of a character.

    Returns one of: 'Basic Latin', 'Latin-1 Supplement', 'Latin Extended-A',
    'Latin Extended-B', or 'Other'.
    """
    cp = ord(char)
    if cp <= 0x007F:
        return "Basic Latin"
    if cp <= 0x00FF:
        return "Latin-1 Supplement"
    if cp <= 0x017F:
        return "Latin Extended-A"
    if cp <= 0x024F:
        return "Latin Extended-B"
    return "Other"


def scrub_non_latin(text: str) -> str:
    """
    Replace any non-Latin Unicode characters with [MARK].

    Spanish 17th-century manuscripts only use Latin script.
    Preserves: Basic Latin, Latin-1 Supplement, Latin Extended A/B,
               standard punctuation, digits, spaces.

    Parameters
    ----------
    text : str
        Raw OCR output text.

    Returns
    -------
    str
        Text with non-Latin characters replaced by [MARK].
    """
    allowed_blocks = {
        "Basic Latin",
        "Latin-1 Supplement",
        "Latin Extended-A",
        "Latin Extended-B",
    }
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        block = _unicode_block(char)
        # Allow characters from Latin blocks or standard punctuation/digits/spaces
        if block in allowed_blocks or unicodedata.category(char) in (
            "Zs",
            "Po",
            "Pd",
            "Nd",
            "Ps",
            "Pe",
        ):
            result.append(char)
        else:
            # Only write [MARK] once for consecutive non-Latin chars
            if not result or result[-1] != "]":
                result.append("[MARK]")
        i += 1
    return "".join(result)


# =============================================================================
# PP-2: Archive Watermark Line Removal
# =============================================================================

_WATERMARK_PATTERNS = [
    re.compile(r"https?://\S+"),  # any URL
    re.compile(r"©|Copyright", re.IGNORECASE),  # copyright symbol
    re.compile(r"Archivos\s+Estatales", re.IGNORECASE),
    re.compile(r"pares\.cultura\.gob\.es"),
    re.compile(r"INQUISICION\s*,?\s*\d{4}"),  # archive shelfmark pattern
    re.compile(r"\bExp\.\s*\d+\b", re.IGNORECASE),  # expedition numbers
    re.compile(r"AHN|AHPG|GPAH", re.IGNORECASE),  # archive acronyms
]


def remove_watermark_lines(text: str) -> str:
    """
    Remove lines that match known archive watermark patterns.

    Parameters
    ----------
    text : str
        Raw OCR output text.

    Returns
    -------
    str
        Text with watermark lines removed.
    """
    lines = text.splitlines()
    clean = [
        line
        for line in lines
        if not any(pat.search(line) for pat in _WATERMARK_PATTERNS)
    ]
    return "\n".join(clean)


# =============================================================================
# PP-5: Repeated-Word Deduplication
# =============================================================================


def deduplicate_line_join_fragments(text: str) -> str:
    """
    Remove word fragments that are immediately followed by the full word.

    Example: "recla reclamaciones" → "reclamaciones"
    Only removes the fragment if it is a strict prefix of the following word.

    Parameters
    ----------
    text : str
        Raw OCR output text.

    Returns
    -------
    str
        Text with duplicate word fragments removed.
    """
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        if (
            i + 1 < len(tokens)
            and tokens[i + 1].startswith(tokens[i])
            and len(tokens[i + 1]) > len(tokens[i])
        ):
            # current token is a prefix of the next — skip it
            i += 1
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


# =============================================================================
# LexicalProcessor Class
# =============================================================================


class LexicalProcessor:
    """
    Applies lexical corrections to OCR output to reduce WER caused by
    inconsistent archaic word spacing (e.g., 'dela' -> 'de la').

    Uses a combination of:
    1. Regex-based heuristics for common 17th-century Spanish concatenations
    2. Optional SymSpell word segmentation (requires Spanish dictionary)

    Post-processing pipeline execution order:
    1. scrub_non_latin()               # PP-1 — remove hallucinated scripts
    2. remove_watermark_lines()        # PP-2 — remove archive artifacts
    3. protect placeholders            # PP-4 guard — before any text manipulation
    4. deduplicate_line_join_fragments() # PP-5
    5. existing spacing corrections    # current LexicalProcessor logic
    6. restore placeholders            # PP-4 restore
    """

    # PP-4: Placeholder patterns to protect
    _PLACEHOLDER_PATTERN = re.compile(
        r"\[\?+\]|\[MARK\]|\[illegible\]|\[MARGINAL NOTE:[^\]]*\]"
    )
    _PLACEHOLDER_SENTINEL = "\x00"  # Sentinel to freeze spaces inside placeholders

    def __init__(self, max_dictionary_edit_distance: int = 0, prefix_length: int = 7):
        """
        Initialize the LexicalProcessor.

        Args:
            max_dictionary_edit_distance: Maximum edit distance for SymSpell lookups.
            prefix_length: Prefix length for SymSpell index.
        """
        self.sym_spell = None
        if SYMSPELL_AVAILABLE:
            self.sym_spell = SymSpell(
                max_dictionary_edit_distance=max_dictionary_edit_distance,
                prefix_length=prefix_length,
            )
            # Note: For production use with 17th-century Spanish texts,
            # load a custom Spanish frequency dictionary:
            # self.sym_spell.load_dictionary("path/to/spanish_frequency.txt")

        # Heuristic patterns for common archaic Spanish word concatenations
        self.corrections = {
            # Preposition + article combinations
            r"\bdela\b": "de la",
            r"\bdelas\b": "de las",
            r"\bdelos\b": "de los",
            r"\bdel\b": "de el",  # Less common, but sometimes needed
            r"\bala\b": "a la",
            r"\balas\b": "a las",
            r"\balos\b": "a los",
            r"\bal\b": "a el",  # Becomes "al" in modern Spanish, but may need splitting
            r"\bconla\b": "con la",
            r"\bconlas\b": "con las",
            r"\bconlos\b": "con los",
            r"\bconel\b": "con el",
            r"\bporla\b": "por la",
            r"\bporlas\b": "por las",
            r"\bporlos\b": "por los",
            r"\bparel\b": "para el",
            r"\bparala\b": "para la",
            r"\bparalas\b": "para las",
            r"\bparalos\b": "para los",
            # Demonstratives with prepositions
            r"\baquel\b": "a quel",  # Context-dependent, may need adjustment
            r"\beste\b": "a este",  # Context-dependent
            r"\bese\b": "a ese",  # Context-dependent
        }

    def _protect_placeholders(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """
        PP-4: Protect placeholders by freezing internal spaces.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        tuple[str, list[tuple[str, str]]]
            Protected text and list of (placeholder, frozen) pairs for restoration.
        """
        frozen_pairs = []

        def freeze(match: re.Match) -> str:
            placeholder = match.group(0)
            frozen = placeholder.replace(" ", self._PLACEHOLDER_SENTINEL)
            frozen_pairs.append((placeholder, frozen))
            return frozen

        protected = self._PLACEHOLDER_PATTERN.sub(freeze, text)
        return protected, frozen_pairs

    def _restore_placeholders(
        self, text: str, frozen_pairs: list[tuple[str, str]]
    ) -> str:
        """
        PP-4: Restore placeholders by unfreezing internal spaces.

        Parameters
        ----------
        text : str
            Text with frozen placeholders.
        frozen_pairs : list[tuple[str, str]]
            List of (placeholder, frozen) pairs from protection step.

        Returns
        -------
        str
            Text with restored placeholders.
        """
        # Restore in reverse order to handle nested cases correctly
        for original, frozen in reversed(frozen_pairs):
            text = text.replace(frozen, original)
        return text

    def process(self, text: str) -> str:
        """
        Apply lexical corrections to the input text.

        Execution order:
        1. scrub_non_latin()               # PP-1 — remove hallucinated scripts
        2. remove_watermark_lines()        # PP-2 — remove archive artifacts
        3. protect placeholders            # PP-4 guard — before any text manipulation
        4. deduplicate_line_join_fragments() # PP-5
        5. existing spacing corrections    # current LexicalProcessor logic
        6. restore placeholders            # PP-4 restore

        Args:
            text: Raw OCR output text.

        Returns:
            Text with corrected word spacing.
        """
        if not text:
            return text

        processed_text = text

        # PP-1: Remove non-Latin characters (hallucinated scripts)
        processed_text = scrub_non_latin(processed_text)

        # PP-2: Remove watermark lines
        processed_text = remove_watermark_lines(processed_text)

        # PP-4: Protect placeholders before any text manipulation
        protected_text, frozen_pairs = self._protect_placeholders(processed_text)

        # PP-5: Remove duplicate word fragments
        deduped_text = deduplicate_line_join_fragments(protected_text)

        # Apply regex-based heuristics for archaic Spanish concatenations
        for pattern, replacement in self.corrections.items():
            deduped_text = re.sub(
                pattern, replacement, deduped_text, flags=re.IGNORECASE
            )

        # Optional: Apply SymSpell word segmentation if a Spanish dictionary is loaded
        # This is disabled by default as it requires a custom dictionary
        # for optimal results with 17th-century Spanish texts.
        # if self.sym_spell and self.sym_spell.word_count > 0:
        #     result = self.sym_spell.word_segmentation(deduped_text)
        #     deduped_text = result.corrected_string

        # PP-4: Restore placeholders
        restored_text = self._restore_placeholders(deduped_text, frozen_pairs)

        return restored_text

    def process_batch(self, texts: list[str]) -> list[str]:
        """
        Apply lexical corrections to a batch of texts.

        Args:
            texts: List of raw OCR output texts.

        Returns:
            List of texts with corrected word spacing.
        """
        return [self.process(text) for text in texts]
