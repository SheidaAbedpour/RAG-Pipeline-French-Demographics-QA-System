import re
import unicodedata


class TextCleaner:
    """Handles text cleaning and normalization"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Comprehensive text cleaning"""
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)

        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€', '"')
        text = text.replace('â€"', '–')
        text = text.replace('â€"', '—')

        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)

        # Clean up sentence boundaries
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)

        return text.strip()
