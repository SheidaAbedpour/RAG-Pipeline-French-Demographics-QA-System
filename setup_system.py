"""
Complete system setup script for France RAG Pipeline.
Run this script to set up everything automatically!
"""
import subprocess
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    """Print setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🇫🇷  FRANCE RAG PIPELINE - COMPLETE SYSTEM SETUP  🇫🇷              ║
║                                                                  ║
║  This script will automatically:                                 ║
║  ✅ Check your environment and API key                           ║
║  ✅ Process Britannica data (scraping + cleaning)                ║
║  ✅ Create vector embeddings                                     ║
║  ✅ Test the complete system                                     ║
║  ✅ Provide you with working RAG API                             ║
║                                                                  ║
║  Time required: ~3-5 minutes                                     ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def check_prerequisites():
    """Check system prerequisites."""
    print("🔍 Checking prerequisites...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}")

    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    print("✅ requirements.txt found")

    # Check API key
    api_key = os.getenv('TOGETHER_API_KEY')
    if not api_key:
        print("❌ TOGETHER_API_KEY not set")
        print("💡 Please set your API key:")
        print("   export TOGETHER_API_KEY='your_api_key_here'")
        print("   Get it from: https://api.together.xyz/settings/api-keys")
        return False
    print(f"✅ API key configured ({api_key[:10]}...)")

    return True


def run_command(command, description, show_output=True):
    """Run a command with error handling."""
    print(f"\n🔄 {description}...")
    print(f"💻 Command: {command}")

    try:
        if show_output:
            # Show output in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Print output line by line
            for line in process.stdout:
                print(f"   {line.rstrip()}")

            process.wait()
            return_code = process.returncode
        else:
            # Run silently
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )
            return_code = result.returncode
            if return_code != 0:
                print(f"   Error: {result.stderr}")

        if return_code == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED (exit code: {return_code})")
            return False

    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def install_dependencies():
    """Install required packages."""
    print("\n📦 INSTALLING DEPENDENCIES")
    print("=" * 50)

    # Upgrade pip first
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip",
        show_output=False
    )

    # Install requirements
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements",
        show_output=False
    )


def process_data():
    """Run data processing pipeline."""
    print("\n📡 DATA PROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Data preprocessing
    success = run_command(
        f"{sys.executable} scripts/data_preprocessing.py",
        "Processing Britannica data (scraping + cleaning)"
    )

    if not success:
        return False

    # Step 2: Create embeddings
    success = run_command(
        f"{sys.executable} scripts/create_embeddings.py",
        "Creating vector embeddings"
    )

    return success


def test_system():
    """Test the complete system."""
    print("\n🧪 SYSTEM TESTING")
    print("=" * 50)

    # Start API server in background
    print("🚀 Starting API server...")

    try:
        # Import here to ensure dependencies are installed
        import threading
        import requests

        def start_api():
            run_command(
                f"{sys.executable} scripts/run_api.py",
                "Starting API server",
                show_output=False
            )

        # Start server in background thread
        server_thread = threading.Thread(target=start_api, daemon=True)
        server_thread.start()

        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(10)

        # Test the server
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running!")

                # Run system tests
                success = run_command(
                    f"{sys.executable} test_system.py",
                    "Running system tests"
                )

                return success
            else:
                print("❌ API server not responding properly")
                return False

        except Exception as e:
            print(f"❌ Cannot connect to API server: {e}")
            return False

    except ImportError:
        print("⚠️ Cannot run automatic testing (missing dependencies)")
        print("💡 Manual testing:")
        print("   1. python scripts/run_api.py")
        print("   2. python test_system.py")
        return True


def show_next_steps():
    """Show next steps to the user."""
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("Your France RAG system is ready to use!")

    print("\n🚀 Quick Start Commands:")
    print("   # Start the API server")
    print("   python scripts/run_api.py")
    print("")
    print("   # Test the system")
    print("   python test_system.py")
    print("")
    print("   # Launch Streamlit UI (bonus +10 points!)")
    print("   python scripts/setup_and_run_app.py")

    print("\n🌐 API Endpoints:")
    print("   Health: http://localhost:8000/health")
    print("   Docs: http://localhost:8000/docs")
    print("   Retrieve: POST http://localhost:8000/retrieve")
    print("   Generate: POST http://localhost:8000/generate")

    print("\n🧪 Quick Test:")
    print("   curl http://localhost:8000/health")

    print("\n📚 Example API Usage:")
    print("""   curl -X POST "http://localhost:8000/generate" \\
     -H "Content-Type: application/json" \\
     -d '{"query": "What are the main mountain ranges in France?"}'""")

    print("\n✨ Your RAG pipeline includes:")
    print("   ✅ Data scraping from Britannica")
    print("   ✅ Text cleaning and chunking")
    print("   ✅ TF-IDF vector embeddings")
    print("   ✅ Hybrid retrieval (vector + metadata)")
    print("   ✅ LLM generation with TogetherAI")
    print("   ✅ FastAPI with proper schemas")
    print("   ✅ Streamlit UI (bonus feature)")


def main():
    """Main setup function."""
    print_banner()

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return False

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        return False

    # Process data
    if not process_data():
        print("\n❌ Data processing failed.")
        return False

    # Test system (optional)
    print("\n🤔 Would you like to run automatic system tests? (y/n): ", end="")
    try:
        choice = input().lower().strip()
        if choice in ['y', 'yes']:
            test_success = test_system()
            if not test_success:
                print("⚠️ Some tests failed, but the system should still work.")
    except KeyboardInterrupt:
        print("\n⏭️ Skipping tests...")

    # Show next steps
    show_next_steps()

    print("\n🎯 Ready for your class presentation!")
    print("Good luck with your RAG project! 🍀")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user.")
        print("💡 You can resume by running individual scripts:")
        print("   python scripts/data_preprocessing.py")
        print("   python scripts/create_embeddings.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)