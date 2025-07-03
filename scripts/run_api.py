import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def main():
    # Set default environment variables
    os.environ.setdefault('DATA_DIR', 'data')
    os.environ.setdefault('USE_SIMPLE_RETRIEVER', 'true')
    os.environ.setdefault('HOST', '0.0.0.0')
    os.environ.setdefault('PORT', '8000')
    os.environ.setdefault('LOG_LEVEL', 'info')

    # Check if API key is set
    if not os.getenv('TOGETHER_API_KEY'):
        print("❌ TOGETHER_API_KEY environment variable is required!")
        print("Get your API key from: https://api.together.xyz/settings/api-keys")
        print("Then set it: export TOGETHER_API_KEY=your_api_key_here")
        return

    # Check if data directory exists
    data_dir = Path(os.getenv('DATA_DIR'))
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run data processing and embeddings first!")
        return

    print("🚀 Starting France RAG API...")
    print(f"📁 Data directory: {data_dir}")
    print(f"🌐 Host: {os.getenv('HOST')}:{os.getenv('PORT')}")
    print(f"📚 Using simple retriever: {os.getenv('USE_SIMPLE_RETRIEVER')}")
    print(f"🔑 API key: {'✅ Set' if os.getenv('TOGETHER_API_KEY') else '❌ Not set'}")

    # Run the API
    uvicorn.run(
        "main:app",
        host=os.getenv('HOST'),
        port=int(os.getenv('PORT')),
        reload=os.getenv('RELOAD', 'false').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL')
    )


if __name__ == "__main__":
    main()