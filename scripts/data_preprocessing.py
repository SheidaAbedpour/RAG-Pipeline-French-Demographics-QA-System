import sys
import os
import json
import argparse
from pathlib import Path
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

from src.data.data_processor import DataProcessor
from src.data.document_chunker import DocumentChunker
from src.data.text_cleaner import TextCleaner


def main():
    parser = argparse.ArgumentParser(description='Process Britannica France Land data')
    parser.add_argument('--output-dir',
                        default=os.getenv("PROCESS_DATA_DIR"),
                        help='Output directory for processed data')
    parser.add_argument('--chunking-strategies',
                        nargs='+',
                        default=['fixed', 'sentence', 'semantic'],
                        help='Chunking strategies to use')
    parser.add_argument('--chunk-size',
                        type=int,
                        default=512,
                        help='Default chunk size for fixed chunking')
    parser.add_argument('--overlap',
                        type=int,
                        default=50,
                        help='Overlap size for chunking')
    parser.add_argument('--delay',
                        type=float,
                        default=1.0,
                        help='Delay between requests (seconds)')
    parser.add_argument('--test-mode',
                        default=False,
                        action='store_true',
                        help='Run in test mode (process only first 2 URLs)')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize processor
    processor = DataProcessor(output_dir=args.output_dir)
    processor.scraper.delay_seconds = args.delay

    # Test mode - process only first 2 URLs
    if args.test_mode:
        logger.info("Running in test mode - processing first 2 URLs only")
        original_urls = processor.scraper.urls.copy()
        test_urls = dict(list(original_urls.items())[:2])
        processor.scraper.urls = test_urls

    try:
        # Process all data
        results = processor.process_all_data(chunking_strategies=args.chunking_strategies)

        # Print summary
        print("\n" + "=" * 60)
        print("DATA PROCESSING SUMMARY")
        print("=" * 60)

        print(f"Raw pages scraped: {len(results['raw_pages'])}")
        print(f"Processed pages: {len(results['processed_pages'])}")

        for strategy, chunks in results['chunks'].items():
            print(f"\n{strategy.upper()} CHUNKING:")
            print(f"  Total chunks: {len(chunks)}")
            print(f"  Average chunk size: {results['statistics'][strategy]['avg_chunk_size']:.1f} tokens")
            print(f"  Min chunk size: {results['statistics'][strategy]['min_chunk_size']} tokens")
            print(f"  Max chunk size: {results['statistics'][strategy]['max_chunk_size']} tokens")

        print(f"\nSections processed: {results['statistics']['fixed']['sections']}")
        print(f"Subsections found: {len(results['statistics']['fixed']['subsections'])}")

        print(f"\nFiles saved to: {args.output_dir}/")
        print("- raw/britannica_raw.json")
        print("- processed/chunks_*.json")
        print("- processed/processing_stats.json")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)


def analyze_chunks():
    """Analyze and compare different chunking strategies"""
    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY ANALYSIS")
    print("=" * 60)

    strategies = ['fixed', 'sentence', 'semantic']

    for strategy in strategies:
        chunk_file = f"data/processed/chunks_{strategy}.json"
        if os.path.exists(chunk_file):
            with open(chunk_file, 'r') as f:
                chunks = json.load(f)

            print(f"\n{strategy.upper()} CHUNKING DETAILED ANALYSIS:")
            print(f"  Total chunks: {len(chunks)}")

            # Analyze chunk sizes
            sizes = [chunk['chunk_size'] for chunk in chunks]
            print(f"  Chunk size distribution:")
            print(f"    Mean: {sum(sizes) / len(sizes):.1f}")
            print(f"    Min: {min(sizes)}")
            print(f"    Max: {max(sizes)}")

            # Analyze by section
            sections = {}
            for chunk in chunks:
                section = chunk['section']
                if section not in sections:
                    sections[section] = 0
                sections[section] += 1

            print(f"  Chunks per section:")
            for section, count in sorted(sections.items()):
                print(f"    {section}: {count}")

            # Show sample chunks
            print(f"  Sample chunks:")
            for i in range(min(2, len(chunks))):
                chunk = chunks[i]
                preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
                print(f"    [{chunk['chunk_id']}]: {preview}")


def test_chunking_parameters():
    """Test different chunking parameters"""
    print("\n" + "=" * 60)
    print("TESTING CHUNKING PARAMETERS")
    print("=" * 60)

    # Load sample text
    if os.path.exists("data/raw/britannica_raw.json"):
        with open("data/raw/britannica_raw.json", 'r') as f:
            raw_data = json.load(f)

        if raw_data:
            sample_text = raw_data[0]['content'][:2000]  # Use first 2000 chars

            chunker = DocumentChunker()

            # Test different chunk sizes
            chunk_sizes = [256, 512, 1024]
            overlaps = [25, 50, 100]

            for chunk_size in chunk_sizes:
                for overlap in overlaps:
                    chunks = chunker.fixed_length_chunking(sample_text, chunk_size, overlap)
                    print(f"Chunk size: {chunk_size}, Overlap: {overlap} -> {len(chunks)} chunks")
    else:
        print("No raw data found. Run data processing first.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_chunks()
    elif len(sys.argv) > 1 and sys.argv[1] == "test-params":
        test_chunking_parameters()
    else:
        main()
