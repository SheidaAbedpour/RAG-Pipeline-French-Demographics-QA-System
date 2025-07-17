import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from src.data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Britannica France data')

    parser.add_argument(
        '--chunking-strategies',
        nargs='+',
        default=['fixed', 'sentence', 'semantic'],
        choices=['fixed', 'sentence', 'semantic'],
        help='Chunking strategies to use'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests (seconds)'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Process only first 2 URLs for testing'
    )

    return parser.parse_args()


def print_summary(results: dict):
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("DATA PROCESSING SUMMARY")
    print("=" * 60)

    print(f"📄 Raw pages scraped: {len(results['raw_pages'])}")
    print(f"🧹 Processed pages: {len(results['processed_pages'])}")

    for strategy, chunks in results['chunks'].items():
        print(f"\n📚 {strategy.upper()} CHUNKING:")
        stats = results['statistics'][strategy]
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Average chunk size: {stats['avg_chunk_size']:.1f} tokens")
        print(f"  Min chunk size: {stats['min_chunk_size']} tokens")
        print(f"  Max chunk size: {stats['max_chunk_size']} tokens")

    sections = results['statistics']['fixed']['sections']
    subsections = results['statistics']['fixed']['subsections']

    print(f"\n📋 Content Structure:")
    print(f"  Sections: {len(sections)}")
    print(f"  Subsections: {len(subsections)}")

    print(f"\n💾 Files saved to: {config.data_dir}/")
    print("  ├── raw/britannica_raw.json")
    print("  ├── processed/chunks_*.json")
    print("  └── processed/processing_stats.json")


def main():
    """Main execution function."""
    try:
        args = parse_arguments()

        print("🚀 Starting France Geography Data Processing...")
        print(f"📁 Output directory: {config.data_dir}")
        print(f"📊 Chunking strategies: {args.chunking_strategies}")
        print(f"⏱️ Request delay: {args.delay}s")

        if args.test_mode:
            print("🧪 Running in test mode (first 2 URLs only)")

        # Initialize processor
        processor = DataProcessor(str(config.data_dir))
        processor.scraper.delay_seconds = args.delay

        # Test mode - process only first 2 URLs
        if args.test_mode:
            original_urls = processor.scraper.urls.copy()
            test_urls = dict(list(original_urls.items())[:2])
            processor.scraper.urls = test_urls
            logger.info(f"Test mode: processing {len(test_urls)} URLs")

        # Process all data
        print("\n📡 Scraping Britannica pages...")
        results = processor.process_all_data(
            chunking_strategies=args.chunking_strategies
        )

        # Print summary
        print_summary(results)

        print("\n✅ Data processing completed successfully!")
        print("🔄 Next step: Run 'python scripts/create_embeddings.py'")

    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n❌ Processing failed!")
        print("💡 Check your internet connection and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()
