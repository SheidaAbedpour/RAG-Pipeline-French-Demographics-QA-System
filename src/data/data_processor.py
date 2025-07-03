import numpy as np
from typing import List, Dict
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.data.britannica_scraper import BritannicaScraper
from src.data.text_cleaner import TextCleaner
from src.data.document_chunker import DocumentChunker


class DataProcessor:
    """Main data processing pipeline"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.scraper = BritannicaScraper()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()

        # Create output directories
        import os
        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        os.makedirs(f"{output_dir}/processed", exist_ok=True)

    def process_all_data(self, chunking_strategies: List[str] = None) -> Dict:
        """Complete data processing pipeline"""
        if chunking_strategies is None:
            chunking_strategies = ['fixed', 'sentence', 'semantic']

        # Step 1: Scrape all pages
        logger.info("Starting data scraping...")
        raw_pages = self.scraper.scrape_all_pages()

        # Save raw data
        with open(f"{self.output_dir}/raw/britannica_raw.json", 'w') as f:
            json.dump(raw_pages, f, indent=2)

        # Step 2: Clean and process text
        logger.info("Cleaning and processing text...")
        processed_pages = []
        for page in raw_pages:
            processed_page = page.copy()
            processed_page['content'] = self.cleaner.clean_text(page['content'])

            # Clean subsections
            cleaned_subsections = []
            for subsection in page.get('subsections', []):
                cleaned_subsection = subsection.copy()
                cleaned_subsection['content'] = self.cleaner.clean_text(subsection['content'])
                cleaned_subsections.append(cleaned_subsection)
            processed_page['subsections'] = cleaned_subsections

            processed_pages.append(processed_page)

        # Step 3: Create chunks with different strategies
        all_chunks = {}

        for strategy in chunking_strategies:
            logger.info(f"Creating chunks with {strategy} strategy...")
            strategy_chunks = []

            chunk_params = {
                'fixed': {'chunk_size': 512, 'overlap': 50},
                'sentence': {'max_chunk_size': 512, 'overlap_sentences': 1},
                'semantic': {'max_chunk_size': 512}
            }

            for page in processed_pages:
                page_chunks = self.chunker.create_document_chunks(
                    page, chunking_method=strategy, **chunk_params[strategy]
                )
                strategy_chunks.extend(page_chunks)

            all_chunks[strategy] = strategy_chunks

            # Save chunks
            chunks_data = []
            for chunk in strategy_chunks:
                chunks_data.append({
                    'text': chunk.text,
                    'metadata': chunk.metadata,
                    'chunk_id': chunk.chunk_id,
                    'source_url': chunk.source_url,
                    'section': chunk.section,
                    'subsection': chunk.subsection,
                    'chunk_index': chunk.chunk_index,
                    'chunk_size': chunk.chunk_size,
                    'overlap_size': chunk.overlap_size
                })

            with open(f"{self.output_dir}/processed/chunks_{strategy}.json", 'w') as f:
                json.dump(chunks_data, f, indent=2)

        # Step 4: Generate processing statistics
        stats = self._generate_statistics(all_chunks)

        with open(f"{self.output_dir}/processed/processing_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("Data processing complete!")
        return {
            'raw_pages': raw_pages,
            'processed_pages': processed_pages,
            'chunks': all_chunks,
            'statistics': stats
        }

    def _generate_statistics(self, all_chunks: Dict) -> Dict:
        """Generate processing statistics"""
        stats = {}

        for strategy, chunks in all_chunks.items():
            strategy_stats = {
                'total_chunks': len(chunks),
                'avg_chunk_size': np.mean([chunk.chunk_size for chunk in chunks]),
                'min_chunk_size': min([chunk.chunk_size for chunk in chunks]),
                'max_chunk_size': max([chunk.chunk_size for chunk in chunks]),
                'sections': list(set([chunk.section for chunk in chunks])),
                'subsections': list(set([chunk.subsection for chunk in chunks if chunk.subsection]))
            }
            stats[strategy] = strategy_stats

        return stats
