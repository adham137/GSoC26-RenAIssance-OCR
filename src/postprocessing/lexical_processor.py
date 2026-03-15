"""
Lexical Post-Processing for OCR Output.

Applies word segmentation and archaic Spanish spacing corrections to reduce
Word Error Rate (WER) in historical document transcription.
"""

import re

try:
    from symspellpy import SymSpell
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False


class LexicalProcessor:
    """
    Applies lexical corrections to OCR output to reduce WER caused by 
    inconsistent archaic word spacing (e.g., 'dela' -> 'de la').
    
    Uses a combination of:
    1. Regex-based heuristics for common 17th-century Spanish concatenations
    2. Optional SymSpell word segmentation (requires Spanish dictionary)
    """
    
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
                prefix_length=prefix_length
            )
            # Note: For production use with 17th-century Spanish texts,
            # load a custom Spanish frequency dictionary:
            # self.sym_spell.load_dictionary("path/to/spanish_frequency.txt")
        
        # Heuristic patterns for common archaic Spanish word concatenations
        self.corrections = {
            # Preposition + article combinations
            r'\bdela\b': 'de la',
            r'\bdelas\b': 'de las',
            r'\bdelos\b': 'de los',
            r'\bdel\b': 'de el',  # Less common, but sometimes needed
            r'\bala\b': 'a la',
            r'\balas\b': 'a las',
            r'\balos\b': 'a los',
            r'\bal\b': 'a el',    # Becomes "al" in modern Spanish, but may need splitting
            r'\bconla\b': 'con la',
            r'\bconlas\b': 'con las',
            r'\bconlos\b': 'con los',
            r'\bconel\b': 'con el',
            r'\bporla\b': 'por la',
            r'\bporlas\b': 'por las',
            r'\bporlos\b': 'por los',
            r'\bparel\b': 'para el',
            r'\bparala\b': 'para la',
            r'\bparalas\b': 'para las',
            r'\bparalos\b': 'para los',
            # Demonstratives with prepositions
            r'\baquel\b': 'a quel',  # Context-dependent, may need adjustment
            r'\beste\b': 'a este',   # Context-dependent
            r'\bese\b': 'a ese',     # Context-dependent
        }
    
    def process(self, text: str) -> str:
        """
        Apply lexical corrections to the input text.
        
        Args:
            text: Raw OCR output text.
            
        Returns:
            Text with corrected word spacing.
        """
        if not text:
            return text
        
        processed_text = text
        
        # Apply regex-based heuristics for archaic Spanish concatenations
        for pattern, replacement in self.corrections.items():
            processed_text = re.sub(
                pattern, 
                replacement, 
                processed_text, 
                flags=re.IGNORECASE
            )
        
        # Optional: Apply SymSpell word segmentation if a Spanish dictionary is loaded
        # This is disabled by default as it requires a custom dictionary
        # for optimal results with 17th-century Spanish texts.
        # if self.sym_spell and self.sym_spell.word_count > 0:
        #     result = self.sym_spell.word_segmentation(processed_text)
        #     processed_text = result.corrected_string
        
        return processed_text
    
    def process_batch(self, texts: list[str]) -> list[str]:
        """
        Apply lexical corrections to a batch of texts.
        
        Args:
            texts: List of raw OCR output texts.
            
        Returns:
            List of texts with corrected word spacing.
        """
        return [self.process(text) for text in texts]
