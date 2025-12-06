# text_context.py - Text context extraction utilities
import logging
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from colors import Colors
except ImportError:
    class Colors:
        MAGENTA = ""


class TextContext:
    """
    Extracts meaningful text segments (contexts) from a given string.
    Identifies substrings that end with predefined split tokens.
    """
    
    def __init__(self, split_tokens: Optional[Set[str]] = None) -> None:
        """
        Initialize the TextContext processor.
        
        Args:
            split_tokens: Set of strings that mark end-of-context.
        """
        if split_tokens is None:
            # Default split tokens including Vietnamese punctuation
            default_splits: Set[str] = {
                ".", "!", "?", ",", ";", ":", "\n", "-",
                "。", "、",  # Japanese/Chinese
                "…"  # Ellipsis
            }
            self.split_tokens: Set[str] = default_splits
        else:
            self.split_tokens: Set[str] = set(split_tokens)

    def get_context(
        self,
        txt: str,
        min_len: int = 6,
        max_len: int = 120,
        min_alnum_count: int = 10
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find the shortest valid context at the beginning of input text.
        
        Args:
            txt: Input string to extract context from.
            min_len: Minimum length for extracted context.
            max_len: Maximum length to search.
            min_alnum_count: Minimum alphanumeric characters required.
            
        Returns:
            Tuple of (context_string, remaining_string) or (None, None).
        """
        alnum_count = 0

        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1

            # Check if current character is a potential context end
            if char in self.split_tokens:
                # Check if criteria are met
                if i >= min_len and alnum_count >= min_alnum_count:
                    context_str = txt[:i]
                    remaining_str = txt[i:]
                    logger.info(f"🧠 {Colors.MAGENTA}Context found after char {i}: {context_str}")
                    return context_str, remaining_str

        # No suitable context found
        return None, None

