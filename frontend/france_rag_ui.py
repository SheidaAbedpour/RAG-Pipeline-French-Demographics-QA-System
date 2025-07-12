"""
France RAG Explorer - Fixed UI with proper text visibility
"""
import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import pandas as pd
from typing import Dict, List, Any

# Page configuration
st.set_page_config(
    page_title="🇫🇷 France RAG Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Fixed CSS with proper text visibility
st.markdown("""
<style>
/* French flag colors */
:root {
    --french-blue: #002654;
    --french-red: #CE1126;
    --light-blue: #4A90E2;
    --light-gray: #F8F9FA;
    --dark-text: #2c3e50;
    --light-text: #34495e;
}

/* Header styling */
.main-header {
    background: linear-gradient(90deg, var(--french-blue) 33%, white 33% 66%, var(--french-red) 66%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.main-header h1 {
    color: white;
    font-size: 3rem;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    font-weight: bold;
}

.main-header p {
    color: white;
    font-size: 1.2rem;
    margin: 0.5rem 0 0 0;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}

/* Chat messages with proper contrast */
.chat-message {
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-left: 5px solid var(--light-blue);
}

.user-message {
    background: linear-gradient(135deg, var(--light-blue), #5DADE2);
    color: white !important;
    margin-left: 10%;
    border-left-color: var(--french-blue);
}

.user-message * {
    color: white !important;
}

.assistant-message {
    background: linear-gradient(135deg, #ffffff, #f8f9fa);
    color: var(--dark-text) !important;
    margin-right: 10%;
    border-left-color: var(--french-red);
    border: 1px solid #e9ecef;
}

.assistant-message * {
    color: var(--dark-text) !important;
}

/* Ensure all text is visible */
.assistant-message p,
.assistant-message div,
.assistant-message span {
    color: var(--dark-text) !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Custom buttons with better colors and contrast */
.stButton > button {
    background: linear-gradient(135deg, #2c3e50, #3498db) !important;
    color: white !important;
    border-radius: 25px !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #34495e, #2980b9) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}

/* Primary button styling (Ask Question button) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    color: white !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #c0392b, #a93226) !important;
}

/* Example question buttons with lighter blue French-inspired colors */
.stButton > button:contains("Mountain"),
.stButton > button:contains("Climate"),
.stButton > button:contains("Rivers"),
.stButton > button:contains("Vegetation"),
.stButton > button:contains("Topography"),
.stButton > button:contains("Agriculture") {
    background: linear-gradient(135deg, #74b9ff, #0984e3) !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 0.6rem 1.5rem !important;
    margin: 0.2rem !important;
    font-size: 0.9rem !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}

.stButton > button:contains("Mountain"):hover,
.stButton > button:contains("Climate"):hover,
.stButton > button:contains("Rivers"):hover,
.stButton > button:contains("Vegetation"):hover,
.stButton > button:contains("Topography"):hover,
.stButton > button:contains("Agriculture"):hover {
    background: linear-gradient(135deg, #0984e3, #2d3436) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2) !important;
}

/* Sidebar buttons */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #8e44ad, #7d3c98) !important;
    color: white !important;
    border-radius: 15px !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #7d3c98, #6c3483) !important;
}

/* Status indicators */
.status-online { 
    color: #28a745 !important; 
    font-weight: bold; 
}

.status-offline { 
    color: #dc3545 !important; 
    font-weight: bold; 
}

/* Source cards with proper contrast */
.source-card {
    background: #ffffff;
    color: var(--dark-text) !important;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border-left: 3px solid var(--french-blue);
    border: 1px solid #dee2e6;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.source-card * {
    color: var(--dark-text) !important;
}

/* Fix Streamlit default text colors */
.stMarkdown p,
.stMarkdown div,
.stText {
    color: var(--dark-text) !important;
}

/* Ensure expander content is visible */
.streamlit-expanderContent {
    background-color: white !important;
    color: var(--dark-text) !important;
}

.streamlit-expanderContent * {
    color: var(--dark-text) !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Force text visibility in all containers */
[data-testid="stMarkdownContainer"] p {
    color: var(--dark-text) !important;
}

/* Fix metric displays */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_status' not in st.session_state:
        st.session_state.api_status = None

initialize_session_state()

# API Functions
@st.cache_data(ttl=30)
def check_api_status():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return False, {"status": "offline", "error": str(e)}

@st.cache_data(ttl=60)
def get_available_sections():
    try:
        response = requests.get(f"{API_BASE_URL}/sections", timeout=5)
        if response.status_code == 200:
            return response.json().get("sections", [])
        return []
    except:
        return []

def call_generate_api(query: str, k: int = 5, section_filter: str = None, temperature: float = 0.3):
    payload = {
        "query": query,
        "k": k,
        "temperature": temperature,
        "max_tokens": 512
    }
    if section_filter and section_filter != "All":
        payload["section_filter"] = section_filter

    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# UI Components
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🇫🇷 France Geography Explorer</h1>
        <p>Découvrez la géographie française avec l'intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)

def render_api_status():
    is_online, health_data = check_api_status()

    if is_online:
        st.success("🟢 **API Server Online** - Ready to explore French geography!")
        with st.expander("🔍 System Status Details"):
            components = health_data.get("components", {})
            for component, status in components.items():
                if "healthy" in status:
                    st.success(f"✅ {component.title()}: {status}")
                else:
                    st.error(f"❌ {component.title()}: {status}")
    else:
        st.error("🔴 **API Server Offline**")
        st.error(f"Error: {health_data.get('error', 'Unknown error')}")
        st.info("💡 **To start the server:** `python scripts/run_api.py`")

    return is_online

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🗺️ Navigation")

        page = st.selectbox(
            "Choose a section:",
            ["🏠 Home & Chat", "📊 Analytics", "💡 Examples"],
            key="page_selector"
        )

        st.markdown("---")
        st.markdown("## ⚙️ Search Settings")

        sections = get_available_sections()
        section_options = ["All"] + sections
        section_filter = st.selectbox("📂 Filter by Section:", section_options, key="section_filter")

        num_results = st.slider("📊 Number of Results:", min_value=1, max_value=10, value=5, key="num_results")
        temperature = st.slider("🌡️ AI Creativity:", min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="temperature")

        st.markdown("---")
        st.markdown("## 🚀 Quick Actions")
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        return page, section_filter, num_results, temperature

def render_example_queries():
    st.markdown("## 💡 Try These Questions")

    examples = [
        ("🏔️ Mountain Ranges", "What are the main mountain ranges in France?"),
        ("🌦️ Climate Patterns", "How does climate vary across France?"),
        ("🏞️ Rivers & Water", "What are the major rivers in France?"),
        ("🌱 Vegetation", "Describe France's plant and animal life"),
        ("🗺️ Topography", "Tell me about France's topographical features"),
        ("🌾 Agriculture", "What are the main soil types in France?")
    ]

    cols = st.columns(2)
    for i, (label, query) in enumerate(examples):
        with cols[i % 2]:
            if st.button(label, key=f"example_{i}", use_container_width=True):
                return query

    return None

def render_chat_interface(section_filter, num_results, temperature):
    st.markdown("## 💬 Ask About French Geography")

    example_query = render_example_queries()

    query = st.text_area(
        "🔎 Your Question:",
        value=example_query if example_query else "",
        placeholder="e.g., What are the main geographical features of France?",
        height=100,
        key="main_query"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)

    with col2:
        if st.button("🗑️", help="Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask_button and query.strip():
        with st.spinner("🔮 AI is thinking..."):
            success, result = call_generate_api(query, num_results, section_filter, temperature)
            if success:
                st.session_state.chat_history.append({
                    'query': query,
                    'answer': result.get('answer', ''),
                    'sources': result.get('sources', []),
                    'metadata': result.get('metadata', {}),
                    'timestamp': datetime.now()
                })
            else:
                st.error(f"❌ Generation failed: {result}")

        st.rerun()

    # Display chat history with proper styling
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("## 💭 Conversation History")

        for chat in reversed(st.session_state.chat_history):
            # User query
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🗣️ You:</strong> {chat['query']}
                <br><small>🕒 {chat['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)

            # AI response with proper text visibility
            answer_text = chat['answer'].replace('\n', '<br>')
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 AI Assistant:</strong><br><br>
                <div style="color: #2c3e50 !important; font-size: 16px; line-height: 1.6;">
                    {answer_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sources
            sources = chat.get('sources', [])
            if sources:
                with st.expander(f"📚 Sources ({len(sources)})"):
                    for j, source in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>📄 Source {j+1}: {source['section']}</strong>
                            <br><strong>🎯 Relevance:</strong> {source['score']:.1%}
                            <br><strong>📝 Content:</strong> {source['text'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)

def render_analytics():
    st.markdown("## 📊 System Analytics")

    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Requests", metrics.get("total_requests", 0))
            with col2:
                st.metric("Generation Requests", metrics.get("generation_requests", 0))
            with col3:
                uptime = metrics.get("uptime_seconds", 0) / 3600
                st.metric("Uptime", f"{uptime:.1f}h")
        else:
            st.error("Failed to load metrics")
    except:
        st.error("Cannot connect to API server")

def render_examples_page():
    st.markdown("## 💡 Examples & Documentation")

    tab1, tab2 = st.tabs(["🔍 Query Examples", "📖 User Guide"])

    with tab1:
        st.markdown("### 🌟 Effective Query Examples")

        categories = {
            "🏔️ Physical Geography": [
                "What are the main mountain ranges in France?",
                "Describe the topographical features of France"
            ],
            "🌦️ Climate & Weather": [
                "How does climate vary across France?",
                "What are France's climate patterns?"
            ],
            "🏞️ Water Systems": [
                "What are the major rivers in France?",
                "Tell me about French drainage systems"
            ]
        }

        for category, examples in categories.items():
            st.markdown(f"**{category}**")
            for example in examples:
                st.write(f"• {example}")

    with tab2:
        st.markdown("### 📖 How to Use France RAG Explorer")
        st.markdown("""
        **🚀 Getting Started:**
        1. Ask questions using natural language
        2. Use the sidebar to filter by sections
        3. Adjust AI creativity with the temperature slider
        
        **🔍 Search Tips:**
        - Be specific: "mountain ranges" vs "mountains"
        - Use geographic terms: "climate patterns", "drainage systems"
        - Ask comparative questions: "How does X compare to Y?"
        """)

# Main Application
def main():
    render_header()

    api_online = render_api_status()
    if not api_online:
        st.stop()

    page, section_filter, num_results, temperature = render_sidebar()

    if page == "🏠 Home & Chat":
        render_chat_interface(section_filter, num_results, temperature)
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "💡 Examples":
        render_examples_page()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🇫🇷 <strong>France Geography Explorer</strong> • Powered by RAG & TogetherAI</p>
        <p><em>Liberté, Égalité, Géographie!</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()