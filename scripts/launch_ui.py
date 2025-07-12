import subprocess
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_api_server():
    """Check if API server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=3)
        return response.status_code == 200
    except:
        return False


def start_api_if_needed():
    """Start API server if not running"""
    if check_api_server():
        print("✅ API server is already running")
        return True

    print("🚀 Starting API server...")
    try:
        import threading

        def run_api():
            subprocess.run([sys.executable, "scripts/run_api.py"],
                           capture_output=True)

        # Start API in background
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        # Wait for it to start
        for i in range(15):
            if check_api_server():
                print("✅ API server started successfully")
                return True
            time.sleep(1)
            print(f"   Waiting... {i + 1}/15")

        print("⚠️ API server might not be ready, but continuing...")
        return True

    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        print("💡 Please start manually: python scripts/run_api.py")
        return False


def launch_ui():
    """Launch the Streamlit UI"""
    ui_file = Path(os.getenv("UI_DIR"))

    if not ui_file.exists():
        print(f"❌ UI file not found: {ui_file}")
        print("💡 Please run: python scripts/setup_and_run_app.py")
        return False

    print("🎉 Launching France RAG Explorer UI...")
    print("🌐 Opening at: http://localhost:8501")
    print("⌨️  Press Ctrl+C to stop")

    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(ui_file),
               "--server.port", "8501", "--server.headless", "true"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 UI closed. Au revoir!")
    except FileNotFoundError:
        print("❌ Streamlit not installed. Please run:")
        print("   pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")


def main():
    print("🇫🇷 France RAG Explorer - UI Launcher")
    print("=" * 40)

    # Check prerequisites
    if not Path("./frontend/france_rag_ui.py").exists():
        print("❌ UI files not found. Setting up...")
        result = subprocess.run([sys.executable, "setup_and_run_app.py"])
        if result.returncode != 0:
            return

    # Start API if needed
    if not start_api_if_needed():
        choice = input("\n⚠️ API server not ready. Continue anyway? (y/n): ")
        if choice.lower() != 'y':
            return

    # Launch UI
    launch_ui()


if __name__ == "__main__":
    main()
