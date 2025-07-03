import re
from dataclasses import dataclass
from typing import List,Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    text: str
    metadata: Dict
    chunk_id: str
    source_url: str
    section: str
    subsection: Optional[str] = None
    chunk_index: int = 0
    chunk_size: int = 0
    overlap_size: int = 0


class DocumentChunker:
    """Handles different chunking strategies"""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or self._default_tokenizer

    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple word-based tokenizer"""
        return text.split()

    def fixed_length_chunking(self, text: str, chunk_size: int = 512,
                              overlap: int = 50) -> List[str]:
        """Fixed-length chunking with overlap"""
        tokens = self.tokenizer(text)
        chunks = []

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            if len(chunk_tokens) > 0:
                chunks.append(' '.join(chunk_tokens))

        return chunks

    def sentence_based_chunking(self, text: str, max_chunk_size: int = 512,
                                overlap_sentences: int = 1) -> List[str]:
        """Sentence-based chunking with overlap"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = len(self.tokenizer(sentence))

            if current_length + sentence_tokens > max_chunk_size and current_chunk:
                # Create chunk
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - overlap_sentences)
                current_chunk = current_chunk[overlap_start:]
                current_length = sum(len(self.tokenizer(s)) for s in current_chunk)

            current_chunk.append(sentence)
            current_length += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def semantic_chunking(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Paragraph-based semantic chunking"""
        # Split by paragraphs (double newlines or similar patterns)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_tokens = len(self.tokenizer(paragraph))

            if current_length + paragraph_tokens > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(paragraph)
            current_length += paragraph_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_document_chunks(self, page_data: Dict, chunking_method: str = 'fixed',
                               **kwargs) -> List[DocumentChunk]:
        """Create DocumentChunk objects from page data"""
        chunks = []

        # Process main content
        main_content = page_data.get('content', '')
        if main_content:
            if chunking_method == 'fixed':
                text_chunks = self.fixed_length_chunking(main_content, **kwargs)
            elif chunking_method == 'sentence':
                text_chunks = self.sentence_based_chunking(main_content, **kwargs)
            elif chunking_method == 'semantic':
                text_chunks = self.semantic_chunking(main_content, **kwargs)
            else:
                raise ValueError(f"Unknown chunking method: {chunking_method}")

            for i, chunk_text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    text=chunk_text,
                    metadata=page_data['metadata'].copy(),
                    chunk_id=f"{page_data['section']}_main_{i}",
                    source_url=page_data['url'],
                    section=page_data['section'],
                    chunk_index=i,
                    chunk_size=len(self.tokenizer(chunk_text)),
                    overlap_size=kwargs.get('overlap', 0)
                )
                chunks.append(chunk)

        # Process subsections
        for subsection in page_data.get('subsections', []):
            subsection_content = subsection.get('content', '')
            if subsection_content:
                if chunking_method == 'fixed':
                    text_chunks = self.fixed_length_chunking(subsection_content, **kwargs)
                elif chunking_method == 'sentence':
                    text_chunks = self.sentence_based_chunking(subsection_content, **kwargs)
                elif chunking_method == 'semantic':
                    text_chunks = self.semantic_chunking(subsection_content, **kwargs)

                for i, chunk_text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        text=chunk_text,
                        metadata=page_data['metadata'].copy(),
                        chunk_id=f"{page_data['section']}_{subsection['title']}_{i}",
                        source_url=page_data['url'],
                        section=page_data['section'],
                        subsection=subsection['title'],
                        chunk_index=i,
                        chunk_size=len(self.tokenizer(chunk_text)),
                        overlap_size=kwargs.get('overlap', 0)
                    )
                    chunks.append(chunk)

        return chunks
