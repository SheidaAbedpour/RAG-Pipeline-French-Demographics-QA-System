import requests
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config.config import config


def test_api_connection():
    """Test if the API server is running."""
    try:
        print("🔌 Testing API connection...")
        response = requests.get(
            f"http://{config.host}:{config.port}/health",
            timeout=5
        )

        if response.status_code == 200:
            health_data = response.json()
            print(f"  ✅ API Status: {health_data['status']}")

            # Check components
            components = health_data.get('components', {})
            for component, status in components.items():
                icon = "✅" if "healthy" in status else "⚠️" if "no_data" in status else "❌"
                print(f"  {icon} {component}: {status}")

            return health_data['status'] == 'healthy'
        else:
            print(f"  ❌ API returned status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("  ❌ Cannot connect to API server")
        print("  💡 Start the server first: python scripts/run_api.py")
        return False
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
        return False


def test_data_files():
    """Check if all required data files exist."""
    print("📁 Checking data files...")

    required_files = [
        (config.processed_dir / "chunks_fixed.json", "Processed chunks"),
        (config.embeddings_dir / "embeddings.npy", "Vector embeddings"),
        (config.embeddings_dir / "metadata.json", "Embedding metadata"),
        (config.embeddings_dir / "vector_store" / "config.json", "Vector store")
    ]

    all_exist = True
    for file_path, description in required_files:
        if file_path.exists():
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            print(f"  ✅ {description} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {description}: Missing")
            all_exist = False

    if not all_exist:
        print("\n💡 Missing files. Run these commands:")
        print("   1. python scripts/data_preprocessing.py")
        print("   2. python scripts/create_embeddings.py")

    return all_exist


def test_retrieval_endpoint():
    """Test the retrieval endpoint with various queries."""
    print("\n🔍 Testing retrieval endpoint...")

    test_queries = [
        {
            "query": "What are the main mountain ranges in France?",
            "expected_sections": ["Land", "The younger mountains"]
        },
        {
            "query": "Tell me about France's climate patterns",
            "expected_sections": ["Climate"]
        },
        {
            "query": "What rivers flow through France?",
            "expected_sections": ["Drainage"]
        },
        {
            "query": "Describe French soil types",
            "expected_sections": ["Soils"]
        }
    ]

    all_passed = True

    for i, test_case in enumerate(test_queries):
        query = test_case["query"]
        expected_sections = test_case["expected_sections"]

        try:
            payload = {"query": query, "k": 5}
            response = requests.post(
                f"http://{config.host}:{config.port}/retrieve",
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                sources = data.get('sources', [])

                if sources:
                    best_source = sources[0]
                    section = best_source['section']
                    score = best_source['score']

                    # Check if we got relevant sections
                    relevant = any(exp in section for exp in expected_sections)
                    icon = "✅" if relevant else "⚠️"

                    print(f"  {icon} Query {i + 1}: {len(sources)} sources")
                    print(f"     Best: {section} (score: {score:.3f})")

                    if not relevant:
                        print(f"     Expected sections: {expected_sections}")
                        all_passed = False
                else:
                    print(f"  ❌ Query {i + 1}: No sources returned")
                    all_passed = False
            else:
                print(f"  ❌ Query {i + 1}: HTTP {response.status_code}")
                all_passed = False

        except Exception as e:
            print(f"  ❌ Query {i + 1}: Error - {e}")
            all_passed = False

    return all_passed


def test_generation_endpoint():
    """Test the generation endpoint."""
    print("\n🤖 Testing generation endpoint...")

    test_queries = [
        "What are the main geographical features of France?",
        "Describe the climate of France"
    ]

    all_passed = True

    for i, query in enumerate(test_queries):
        try:
            payload = {
                "query": query,
                "k": 3,
                "temperature": 0.3,
                "max_tokens": 200
            }

            print(f"  🔄 Query {i + 1}: Generating answer...")
            start_time = time.time()

            response = requests.post(
                f"http://{config.host}:{config.port}/generate",
                json=payload,
                timeout=30
            )

            end_time = time.time()
            duration = end_time - start_time

            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                sources = data.get('sources', [])

                if answer and len(answer) > 50:
                    print(f"  ✅ Query {i + 1}: Generated {len(answer)} chars in {duration:.1f}s")
                    print(f"     Sources used: {len(sources)}")
                    print(f"     Preview: {answer[:100]}...")

                    # Check for source references
                    has_references = any(word in answer.lower() for word in
                                         ['according', 'based on', 'section', 'source'])
                    if has_references:
                        print(f"     ✅ Answer references sources")
                    else:
                        print(f"     ⚠️ Answer doesn't reference sources")
                else:
                    print(f"  ❌ Query {i + 1}: Answer too short ({len(answer)} chars)")
                    all_passed = False
            else:
                error_text = response.text[:200] if response.text else "No error message"
                print(f"  ❌ Query {i + 1}: HTTP {response.status_code}")
                print(f"     Error: {error_text}")
                all_passed = False

        except Exception as e:
            print(f"  ❌ Query {i + 1}: Error - {e}")
            all_passed = False

    return all_passed


def test_metadata_endpoints():
    """Test metadata and utility endpoints."""
    print("\n📊 Testing metadata endpoints...")

    endpoints = [
        ("/sections", "Available sections"),
        ("/metrics", "System metrics")
    ]

    all_passed = True

    for endpoint, description in endpoints:
        try:
            response = requests.get(
                f"http://{config.host}:{config.port}{endpoint}",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()

                if endpoint == "/sections":
                    sections = data.get('sections', [])
                    print(f"  ✅ {description}: {len(sections)} sections")
                    print(f"     Available: {', '.join(sections[:3])}...")

                elif endpoint == "/metrics":
                    total_requests = data.get('total_requests', 0)
                    uptime = data.get('uptime_seconds', 0)
                    print(f"  ✅ {description}: {total_requests} requests, {uptime:.0f}s uptime")

            else:
                print(f"  ❌ {description}: HTTP {response.status_code}")
                all_passed = False

        except Exception as e:
            print(f"  ❌ {description}: Error - {e}")
            all_passed = False

    return all_passed


def run_performance_test():
    """Run a simple performance test."""
    print("\n⚡ Running performance test...")

    query = "What are the main mountain ranges in France?"
    num_requests = 5
    times = []

    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"http://{config.host}:{config.port}/retrieve",
                json={"query": query, "k": 3},
                timeout=10
            )
            end_time = time.time()

            if response.status_code == 200:
                times.append(end_time - start_time)

        except Exception:
            pass

    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        print(f"  📈 Retrieval performance ({len(times)}/{num_requests} successful):")
        print(f"     Average: {avg_time:.3f}s")
        print(f"     Range: {min_time:.3f}s - {max_time:.3f}s")

        if avg_time < 0.5:
            print(f"  ✅ Performance: Excellent")
        elif avg_time < 1.0:
            print(f"  ✅ Performance: Good")
        else:
            print(f"  ⚠️ Performance: Could be better")
    else:
        print(f"  ❌ Performance test failed")


def main():
    """Run complete system test."""
    print("🧪 France RAG System Test Suite")
    print("=" * 50)

    try:
        # Validate config
        config.validate()
        print("✅ Configuration valid")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return

    # Test data files
    if not test_data_files():
        print("\n❌ Data files missing. System test cannot continue.")
        return

    # Test API connection
    if not test_api_connection():
        print("\n❌ API server not accessible. System test cannot continue.")
        return

    # Run API tests
    retrieval_ok = test_retrieval_endpoint()
    generation_ok = test_generation_endpoint()
    metadata_ok = test_metadata_endpoints()

    # Performance test
    run_performance_test()

    # Final summary
    print("\n" + "=" * 50)
    print("📋 SYSTEM TEST SUMMARY")
    print("=" * 50)

    tests = [
        ("Data Files", True),  # Already checked above
        ("API Connection", True),  # Already checked above
        ("Retrieval Endpoint", retrieval_ok),
        ("Generation Endpoint", generation_ok),
        ("Metadata Endpoints", metadata_ok)
    ]

    passed = sum(1 for _, ok in tests if ok)
    total = len(tests)

    for test_name, ok in tests:
        icon = "✅" if ok else "❌"
        print(f"{icon} {test_name}")

    print(f"\n📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✨ Your France RAG system is working perfectly!")
        print("\n🚀 Ready for production!")
        print("💡 Try the Streamlit UI: python scripts/setup_and_run_app.py")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

        if not retrieval_ok:
            print("🔧 Retrieval issues: Check embeddings and vector store")
        if not generation_ok:
            print("🔧 Generation issues: Check TogetherAI API key and model")
        if not metadata_ok:
            print("🔧 Metadata issues: Check API endpoints")

    print("\n🌐 API URLs:")
    print(f"   Health: http://{config.host}:{config.port}/health")
    print(f"   Docs: http://{config.host}:{config.port}/docs")
    print(f"   Test: curl http://{config.host}:{config.port}/health")


if __name__ == "__main__":
    main()
