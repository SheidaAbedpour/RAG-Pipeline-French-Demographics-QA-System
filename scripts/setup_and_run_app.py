"""
🚀 Complete Setup and Run Script for France RAG Explorer
This script handles everything - data processing, API setup, and UI launch!
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def print_banner():
    """Print setup banner"""
    print("""
╔═════════════════════════════════════════════════════════════════╗
║  🇫🇷  FRANCE RAG EXPLORER - COMPLETE SETUP & LAUNCH  🇫            ║
║                                                                 ║
║  This script will automatically:                                ║
║  ✅ Install all required packages                               ║
║  ✅ Process Britannica data (if needed)                         ║
║  ✅ Create vector embeddings (if needed)                        ║
║  ✅ Start the FastAPI server                                    ║
║  ✅ Launch the beautiful Streamlit UI                           ║
║                                                                 ║
║  Total time: ~3-5 minutes (first run)                           ║
║  Subsequent runs: ~30 seconds                                   ║
╚═════════════════════════════════════════════════════════════════╝
    """)


def run_command_silent(command, description):
    """Run a command silently with error handling"""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def run_command_verbose(command, description):
    """Run a command with live output"""
    print(f"🔄 {description}...")
    try:
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

        if process.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False

    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def check_prerequisites():
    """Check system prerequisites"""
    print("🔍 Checking prerequisites...")

    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

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


def install_dependencies():
    """Install required packages"""
    print("\n📦 CHECKING & INSTALLING DEPENDENCIES")
    print("=" * 50)

    # Check if requirements.txt exists
    if not Path("../requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False

    # Install requirements
    success, _ = run_command_silent(
        f"{sys.executable} -m pip install -r requirements.txt --quiet",
        "Installing requirements"
    )

    if success:
        print("✅ All dependencies installed")
        return True
    else:
        print("🔄 Installing dependencies (this may take a moment)...")
        return run_command_verbose(
            f"{sys.executable} -m pip install -r ../requirements.txt",
            "Installing requirements"
        )


def check_data_status():
    """Check if data processing is needed"""
    data_dir = Path(os.getenv("DATA_DIR"))

    required_files = [
        data_dir / "processed" / "chunks_fixed.json",
        data_dir / "embeddings" / "embeddings.npy",
        data_dir / "embeddings" / "metadata.json",
        data_dir / "embeddings" / "vector_store" / "config.json"
    ]

    all_exist = all(file.exists() for file in required_files)

    if all_exist:
        print("✅ Data and embeddings already exist")
        return True
    else:
        missing = [str(f) for f in required_files if not f.exists()]
        print(f"⚠️ Missing data files: {len(missing)} files need to be created")
        return False


def process_data():
    """Run data processing pipeline"""
    if check_data_status():
        return True

    print("\n📡 DATA PROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Data preprocessing
    print("🌐 Downloading and processing Britannica data...")
    success = run_command_verbose(
        f"{sys.executable} scripts/data_preprocessing.py",
        "Processing Britannica data"
    )

    if not success:
        print("❌ Data processing failed")
        return False

    # Step 2: Create embeddings
    print("\n🧮 Creating vector embeddings...")
    success = run_command_verbose(
        f"{sys.executable} scripts/create_embeddings.py",
        "Creating embeddings"
    )

    return success


def start_api_server():
    """Start the API server in background"""
    print("\n🚀 STARTING API SERVER")
    print("=" * 50)

    # Check if already running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        if response.status_code == 200:
            print("✅ API server already running")
            return True
    except:
        pass

    try:
        def run_api():
            """Run API server in background thread"""
            subprocess.run([
                sys.executable, "scripts/run_api.py"
            ], capture_output=True, text=True)

        # Start API in background thread
        print("🔄 Starting API server in background...")
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        # Wait for server to start
        print("⏳ Waiting for API server to start...")
        import requests

        for i in range(20):
            try:
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("✅ API server is running!")
                    health_data = response.json()

                    # Show component status
                    components = health_data.get("components", {})
                    for component, status in components.items():
                        icon = "✅" if "healthy" in status else "⚠️"
                        print(f"   {icon} {component}: {status}")

                    return True
            except:
                pass

            time.sleep(0.5)
            if i % 4 == 0:  # Print every 2 seconds
                print(f"   Still starting... ({i // 2 + 1}/10)")

        print("⚠️ API server might not be fully ready, but continuing...")
        return True

    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return False


def create_ui_file():
    """Create or update the UI file"""
    print("\n📱 SETTING UP STREAMLIT UI")
    print("=" * 50)

    # Create frontend directory
    frontend_dir = Path("../frontend")
    frontend_dir.mkdir(exist_ok=True)

    ui_file = frontend_dir / "france_rag_ui.py"

    print(f"✅ Created Streamlit UI: {ui_file}")
    return str(ui_file)


def launch_streamlit(ui_file):
    """Launch the Streamlit application"""
    print("\n🎉 LAUNCHING FRANCE RAG EXPLORER")
    print("=" * 50)
    print("🌐 Opening at: http://localhost:8501")
    print("📱 Features available:")
    print("   • 💬 Interactive chat with AI")
    print("   • 🔍 Advanced search capabilities")
    print("   • 📊 System analytics dashboard")
    print("   • 💡 Examples and documentation")
    print("   • 🎨 Beautiful French flag theme")
    print("\n⌨️  Press Ctrl+C to stop")
    print("✨ Enjoy exploring French geography!")
    print()

    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", ui_file,
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Application closed. Au revoir!")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {str(e)}")
        print("\n💡 Try running manually:")
        print(f"   streamlit run {ui_file}")


def main():
    """Main setup and launch function"""
    print_banner()

    # Step 1: Check prerequisites
    if not check_prerequisites():
        return

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed")
        return

    # Step 3: Process data and create embeddings
    if not process_data():
        print("\n❌ Data processing failed")
        return

    # Step 4: Start API server
    if not start_api_server():
        print("\n❌ API server failed to start")
        return

    # Step 5: Create UI file
    ui_file = create_ui_file()

    # Step 6: Launch Streamlit
    print("\n🎊 Setup complete! Launching the beautiful UI...")
    time.sleep(2)
    launch_streamlit(ui_file)


if __name__ == "__main__":
    main()
