"""
🚀 France RAG Explorer - Main Launcher
Run this file from the project root to start everything!

Usage: python run_app.py
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
    """Print the application banner"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  🇫🇷  FRANCE RAG EXPLORER - MAIN LAUNCHER  🇫🇷                   ║
║                                                                  ║
║  Starting your complete RAG system...                           ║
║  • FastAPI Backend                                              ║
║  • Beautiful Streamlit UI                                       ║
║  • Real-time geography Q&A                                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def check_prerequisites():
    """Check if the system is ready to run"""
    print("🔍 Checking system prerequisites...")

    issues = []

    # Check API key
    if not os.getenv('TOGETHER_API_KEY'):
        issues.append("❌ TOGETHER_API_KEY not set")
        issues.append("   Set with: export TOGETHER_API_KEY='your_key_here'")
        issues.append("   Get key from: https://api.together.xyz/settings/api-keys")
    else:
        print("✅ API key configured")

    # Check required files
    required_files = [
        Path("data/processed/chunks_fixed.json"),
        Path("data/embeddings/embeddings.npy"),
        Path("data/embeddings/metadata.json"),
        Path("data/embeddings/vector_store/config.json")
    ]

    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))

    if missing_files:
        issues.append("❌ Missing data files:")
        for file_path in missing_files:
            issues.append(f"   - {file_path}")
        issues.append("   Run setup: python scripts/setup_and_run_app.py")
    else:
        print("✅ All data files found")

    # Check UI file
    ui_file = Path("frontend/france_rag_ui.py")
    if not ui_file.exists():
        issues.append(f"❌ UI file missing: {ui_file}")
        issues.append("   Run setup: python scripts/setup_and_run_app.py")
    else:
        print("✅ UI file found")

    if issues:
        print("\n⚠️ Issues found:")
        for issue in issues:
            print(issue)
        return False

    print("✅ All prerequisites satisfied!")
    return True


def start_api_server():
    """Start the FastAPI server in background"""
    print("🚀 Starting FastAPI server...")

    def run_api():
        try:
            subprocess.run([
                sys.executable, "scripts/run_api.py"
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ API server error: {e}")
        except Exception as e:
            print(f"❌ Failed to start API: {e}")

    # Start in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Wait for server to start
    print("⏳ Waiting for API server to initialize...")

    for i in range(15):
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ API server is running!")

                # Show component status
                health_data = response.json()
                components = health_data.get("components", {})
                for component, status in components.items():
                    icon = "✅" if "healthy" in status else "⚠️"
                    print(f"   {icon} {component}: {status}")

                return True
        except:
            pass

        time.sleep(1)
        if i % 3 == 0:
            print(f"   Still starting... ({i // 3 + 1}/5)")

    print("⚠️ API server may not be fully ready, but continuing...")
    return True


def launch_streamlit():
    """Launch the Streamlit UI"""
    print("\n🎨 Launching Streamlit UI...")
    print("🌐 Opening at: http://localhost:8501")
    print("📱 Features available:")
    print("   • 💬 Interactive chat with AI")
    print("   • 📊 System analytics dashboard")
    print("   • 💡 Examples and documentation")
    print("   • 🎨 Beautiful French flag theme")
    print("\n⌨️  Press Ctrl+C to stop both services")
    print("✨ Enjoy exploring French geography!\n")

    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "frontend/france_rag_ui.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user. Au revoir!")
    except FileNotFoundError:
        print("❌ Streamlit not installed. Please run:")
        print("   pip install streamlit")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit error: {e}")
        print("💡 Try running manually:")
        print("   streamlit run frontend/france_rag_ui.py")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")


def show_urls():
    """Show important URLs to the user"""
    print("🔗 Important URLs:")
    print("   🌐 Streamlit UI: http://localhost:8501")
    print("   🔧 FastAPI Docs: http://localhost:8000/docs")
    print("   🏥 Health Check: http://localhost:8000/health")
    print("   📊 API Metrics: http://localhost:8000/metrics")


def quick_test():
    """Run a quick system test"""
    print("\n🧪 Running quick system test...")

    try:
        import requests

        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API health check passed")
        else:
            print("❌ API health check failed")
            return False

        # Test a simple query
        test_payload = {
            "query": "French mountains",
            "k": 2,
            "temperature": 0.3
        }

        response = requests.post(
            "http://localhost:8000/generate",
            json=test_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer_length = len(result.get('answer', ''))
            source_count = len(result.get('sources', []))

            print(f"✅ Generation test passed")
            print(f"   Answer length: {answer_length} characters")
            print(f"   Sources found: {source_count}")
            return True
        else:
            print("❌ Generation test failed")
            return False

    except Exception as e:
        print(f"❌ System test error: {e}")
        return False


def main():
    """Main launcher function"""
    print_banner()

    # Check if everything is ready
    if not check_prerequisites():
        print("\n💡 To fix issues, run the setup script:")
        print("   python scripts/setup_and_run_app.py")
        return

    try:
        # Start API server
        if not start_api_server():
            print("❌ Failed to start API server")
            return

        # Run quick test
        test_passed = quick_test()
        if not test_passed:
            print("⚠️ System test had issues, but UI will still launch")

        # Show URLs
        show_urls()

        # Small delay before launching UI
        print("\n🎊 System ready! Launching UI in 3 seconds...")
        time.sleep(3)

        # Launch Streamlit UI (this will block until user stops)
        launch_streamlit()

    except KeyboardInterrupt:
        print("\n👋 Shutting down France RAG Explorer...")
        print("Thank you for exploring French geography! 🇫🇷")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try running the setup script:")
        print("   python scripts/setup_and_run_app.py")


if __name__ == "__main__":
    main()