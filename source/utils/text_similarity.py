# text_similarity.py - Text similarity comparison utilities
import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class TextSimilarity:
    """
    Compares two text strings and calculates their similarity ratio.
    Supports different comparison strategies: overall, end-focused, or weighted.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.96,
        n_words: int = 5,
        focus: str = 'weighted',
        end_weight: float = 0.7
    ):
        """
        Initialize the TextSimilarity comparator.
        
        Args:
            similarity_threshold: Minimum similarity ratio for texts to be considered similar.
            n_words: Number of words from the end to consider for focused comparison.
            focus: Comparison strategy ('overall', 'end', or 'weighted').
            end_weight: Weight for end similarity in 'weighted' mode.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if not isinstance(n_words, int) or n_words < 1:
            raise ValueError("n_words must be a positive integer")
        if focus not in ['end', 'weighted', 'overall']:
            raise ValueError("focus must be 'end', 'weighted', or 'overall'")
        if not 0.0 <= end_weight <= 1.0:
            raise ValueError("end_weight must be between 0.0 and 1.0")

        self.similarity_threshold = similarity_threshold
        self.n_words = n_words
        self.focus = focus
        self.end_weight = end_weight if focus == 'weighted' else 0.0
        
        # Precompile regex for efficiency
        self._punctuation_regex = re.compile(r'[^\w\s]')
        self._whitespace_regex = re.compile(r'\s+')

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not isinstance(text, str):
            logger.warning(f"📏⚠️ Input is not a string: {type(text)}. Converting to empty string.")
            text = ""
        text = text.lower()
        text = self._punctuation_regex.sub('', text)
        text = self._whitespace_regex.sub(' ', text).strip()
        return text

    def _get_last_n_words_text(self, normalized_text: str) -> str:
        """Extract the last n_words from normalized text."""
        words = normalized_text.split()
        last_words_segment = words[-self.n_words:]
        return ' '.join(last_words_segment)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity ratio between two texts.
        
        Returns:
            Float between 0.0 and 1.0 representing similarity.
        """
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)

        # Both empty -> perfect match
        if not norm_text1 and not norm_text2:
            return 1.0

        matcher = SequenceMatcher(isjunk=None, a=None, b=None, autojunk=False)

        if self.focus == 'overall':
            matcher.set_seqs(norm_text1, norm_text2)
            return matcher.ratio()

        elif self.focus == 'end':
            end_text1 = self._get_last_n_words_text(norm_text1)
            end_text2 = self._get_last_n_words_text(norm_text2)
            matcher.set_seqs(end_text1, end_text2)
            return matcher.ratio()

        elif self.focus == 'weighted':
            # Overall similarity
            matcher.set_seqs(norm_text1, norm_text2)
            sim_overall = matcher.ratio()

            # End similarity
            end_text1 = self._get_last_n_words_text(norm_text1)
            end_text2 = self._get_last_n_words_text(norm_text2)
            matcher.set_seqs(end_text1, end_text2)
            sim_end = matcher.ratio()

            # Weighted average
            weighted_sim = (1 - self.end_weight) * sim_overall + self.end_weight * sim_end
            return weighted_sim

        else:
            logger.error(f"📏💥 Invalid focus mode: {self.focus}")
            raise RuntimeError("Invalid focus mode")

    def are_texts_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts meet the similarity threshold."""
        similarity = self.calculate_similarity(text1, text2)
        return similarity >= self.similarity_threshold

