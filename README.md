# 🇫🇷 France RAG Explorer

**Retrieval-Augmented Generation Pipeline for French Geography**

A complete end-to-end RAG system that scrapes Britannica's France geography content, creates vector embeddings, and provides intelligent Q&A capabilities through a FastAPI backend and beautiful Streamlit UI.

![gif](https://github.com/SheidaAbedpour/RAG-Pipeline/blob/main/docs/ScreenVideo.gif)

## 🎯 Project Overview

This project implements a sophisticated RAG pipeline that:
- Scrapes and processes geographical data about France from Britannica
- Creates vector embeddings using TF-IDF or sentence transformers
- Provides hybrid retrieval combining vector similarity and metadata filtering
- Generates contextual answers using TogetherAI's LLM models
- Exposes clean REST API endpoints
- Includes a Streamlit UI for interactive exploration

## 🗺️ Project Roadmap

```mermaid
graph TB
    A[🚀 Project Start] --> B[📋 Setup & Planning]
    B --> C[🌐 Data Collection]
    C --> D[🧹 Data Processing]
    D --> E[🧮 Embedding Creation]
    E --> F[🔍 Retrieval System]
    F --> G[🤖 Generation System]
    G --> H[📡 API Development]
    H --> I[🧪 Testing & Validation]
    I --> J[🎨 UI Development]
    J --> K[🚀 Deployment]
    K --> L[✅ Project Complete]

    %% Detailed breakdowns
    C --> C1[Britannica Scraping]
    C --> C2[Content Validation]
    
    D --> D1[Text Cleaning]
    D --> D2[Chunking Strategies]
    D --> D3[Metadata Extraction]
    
    E --> E1[TF-IDF Vectors]
    E --> E2[Sentence Transformers]
    
    F --> F1[Vector Store]
    F --> F2[Hybrid Search]
    F --> F3[Metadata Filtering]
    
    G --> G1[Prompt Engineering]
    G --> G2[LLM Integration]
    G --> G3[Context Management]
    
    H --> H1[FastAPI Setup]
    H --> H2[Endpoint Design]
    H --> H3[Error Handling]
    
    I --> I1[Unit Tests]
    I --> I2[Integration Tests]
    I --> I3[Performance Tests]
    
    J --> J1[Streamlit UI]
    J --> J2[Interactive Features]

    %% Styling
    classDef phase fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef milestone fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef bonus fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class A,B,C,D,E,F,G,H,I,K,L phase
    class J,J1,J2 bonus
```

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "🌐 Data Layer"
        A1[Britannica Scraper] --> A2[Text Cleaner]
        A2 --> A3[Document Chunker]
        A3 --> A4[Metadata Extractor]
    end
    
    subgraph "🧮 Embedding Layer"
        B1[TF-IDF Vectorizer]
        B2[Sentence Transformers]
        B3[Vector Store]
    end
    
    subgraph "🔍 Retrieval Layer"
        C1[Cosine Similarity]
        C2[Metadata Filtering]
        C3[Hybrid Retriever]
    end
    
    subgraph "🤖 Generation Layer"
        D1[Prompt Engineer]
        D2[Context Formatter]
        D3[TogetherAI LLM]
        D4[Response Validator]
    end
    
    subgraph "📡 API Layer"
        E1[FastAPI Server]
        E2[Retrieve Endpoint]
        E3[Generate Endpoint]
        E4[Health and Metrics]
    end
    
    subgraph "🎨 Interface Layer"
        F1[Streamlit UI]
        F2[Interactive Chat]
        F3[Analytics Dashboard]
    end
    
    %% Data flow connections
    A4 --> B1
    A4 --> B2
    B1 --> B3
    B2 --> B3
    B3 --> C1
    A4 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    C3 --> E2
    D4 --> E3
    E2 --> E1
    E3 --> E1
    E4 --> E1
    E1 --> F1
    E2 --> F2
    E3 --> F2
    E1 --> F3
    
    %% Layer styling
    classDef dataLayer fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef embeddingLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef retrievalLayer fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef generationLayer fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef apiLayer fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef uiLayer fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    
    class A1,A2,A3,A4 dataLayer
    class B1,B2,B3 embeddingLayer
    class C1,C2,C3 retrievalLayer
    class D1,D2,D3,D4 generationLayer
    class E1,E2,E3,E4 apiLayer
    class F1,F2,F3 uiLayer
```

## 📊 Development Timeline

```mermaid
gantt
    title France RAG Pipeline Development
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Environment Setup    :done, setup, 2024-01-01, 1d
    Project Structure    :done, structure, after setup, 1d
    Dependencies         :done, deps, after structure, 1d
    
    section Phase 2: Data Pipeline
    Web Scraping         :done, scraping, after deps, 2d
    Text Processing      :done, processing, after scraping, 2d
    Chunking Strategies  :done, chunking, after processing, 1d
    
    section Phase 3: Embedding
    TF-IDF Implementation:done, tfidf, after chunking, 1d
    Vector Store         :done, vectorstore, after tfidf, 1d
    Embedding Optimization:done, embedding-opt, after vectorstore, 1d
    
    section Phase 4: Retrieval
    Similarity Search    :done, similarity, after embedding-opt, 1d
    Hybrid Filtering     :done, hybrid, after similarity, 1d
    Retrieval Testing    :done, ret-test, after hybrid, 1d
    
    section Phase 5: Generation
    Prompt Engineering   :done, prompts, after ret-test, 2d
    LLM Integration      :done, llm, after prompts, 1d
    Response Validation  :done, validation, after llm, 1d
    
    section Phase 6: API
    FastAPI Setup        :done, api-setup, after validation, 1d
    Endpoint Development :done, endpoints, after api-setup, 2d
    Error Handling       :done, errors, after endpoints, 1d
    
    section Phase 7: Testing
    Unit Tests           :done, unit, after errors, 1d
    Integration Tests    :done, integration, after unit, 1d
    Performance Tests    :done, performance, after integration, 1d
    
    section Phase 8: UI (Bonus)
    Streamlit Setup      :done, ui-setup, after performance, 1d
    Interactive Features :done, ui-features, after ui-setup, 2d
    UI Polish           :done, ui-polish, after ui-features, 1d
    
    section Phase 9: Deployment
    Documentation        :done, docs, after ui-polish, 1d
    Final Testing        :done, final-test, after docs, 1d
    Deployment           :done, deploy, after final-test, 1d
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TogetherAI API key ([Get one here](https://api.together.xyz/settings/api-keys))

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/SheidaAbedpour/RAG-Pipeline
cd france-rag-explorer

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export TOGETHER_API_KEY="your_api_key_here"
# Or create a .env file:
echo "TOGETHER_API_KEY=your_api_key_here" > .env
```

### 2. Complete Setup (Recommended)

Run the automated setup script that handles everything:

```bash
python setup_system.py
```

This will:
- ✅ Check prerequisites
- ✅ Install dependencies
- ✅ Scrape and process Britannica data
- ✅ Create vector embeddings
- ✅ Test the system
- ✅ Provide usage instructions

### 3. Manual Setup (Alternative)

If you prefer step-by-step setup:

```bash
# Step 1: Process data
python scripts/data_preprocessing.py

# Step 2: Create embeddings
python scripts/create_embeddings.py

# Step 3: Test system
python test_system.py
```

### 4. Start the System

```bash
# Option A: Run complete app (API + UI)
python run_app.py

# Option B: API only
python scripts/run_api.py

# Option C: Full setup + UI
python scripts/setup_and_run_app.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Retrieve Sources
```bash
POST /retrieve
Content-Type: application/json

{
  "query": "What are the main mountain ranges in France?",
  "k": 5,
  "section_filter": "Land",
  "min_score": 0.0
}
```

### Generate Answer
```bash
POST /generate
Content-Type: application/json

{
  "query": "Describe the climate of France",
  "k": 5,
  "temperature": 0.3,
  "max_tokens": 512
}
```

### Example Usage

```bash
# Test the API
curl http://localhost:8000/health

# Get sources
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "French mountains", "k": 3}'

# Generate answer
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main geographical features of France?"}'
```

## 🎯 Key Features & Capabilities

```mermaid
mindmap
  root((France RAG Explorer))
    (🌐 Data Processing)
      [Web Scraping]
        ::icon(fa fa-globe)
        Real-time Britannica data
        Respectful crawling
        Content validation
      [Text Cleaning]
        ::icon(fa fa-broom)
        Unicode normalization
        Encoding fixes
        Noise removal
      [Smart Chunking]
        ::icon(fa fa-cut)
        Fixed-length strategy
        Sentence boundaries
        Semantic splits
    (🧮 Vector Search)
      [TF-IDF Embeddings]
        ::icon(fa fa-calculator)
        5000 features
        1-2 gram range
        Fast inference
      [Hybrid Retrieval]
        ::icon(fa fa-search)
        Vector similarity
        Metadata filtering
        Top-K optimization
    (🤖 AI Generation)
      [LLM Integration]
        ::icon(fa fa-brain)
        Llama 3.3 70B
        TogetherAI API
        Smart prompting
      [Context Management]
        ::icon(fa fa-list)
        4K token context
        Source attribution
        Quality validation
    (📡 API & UI)
      [FastAPI Backend]
        ::icon(fa fa-server)
        REST endpoints
        Pydantic schemas
        Error handling
      [Streamlit Frontend]
        ::icon(fa fa-desktop)
        Interactive chat
        Analytics dashboard
        French theme
```

## 🧱 System Components

### 1. Data Processing 

**Text Cleaning** (`src/data/text_cleaner.py`):
- Unicode normalization (NFKD)
- Whitespace normalization
- Special character handling
- Encoding issue fixes
- Punctuation cleanup

**Document Chunking** (`src/data/document_chunker.py`):
- **Fixed-length**: 512 tokens with 50-token overlap
- **Sentence-based**: Semantic boundaries with overlap
- **Semantic**: Paragraph-based natural splits
- Configurable parameters for optimization

**Metadata Extraction**:
- Section/subsection hierarchies
- Source URLs and domains
- Content type classifications
- Chunk size and overlap tracking

### 2. Retrieval Strategy 

**Vector Embeddings** (`src/embedding.py`):
- **TF-IDF**: 5000 features, 1-2 gram range, English stopwords
- **Sentence Transformers**: all-MiniLM-L6-v2 (384 dimensions)
- Trade-off: TF-IDF for speed, transformers for semantic quality

**Hybrid Retrieval** (`src/retrieval/hybrid_retriever.py`):
- Cosine similarity vector search
- Metadata filtering by section/subsection
- Score thresholding
- Configurable top-K results

**Retrieval Optimization**:
- 3x initial retrieval pool for filtering
- Section-aware result ranking
- Content statistics and availability checks

### 3. Generation Strategy 

**LLM Integration** (`src/generation.py`):
- **Model**: meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
- **Temperature**: 0.3 (balanced creativity/consistency)
- **Max tokens**: 1024 (comprehensive answers)
- **Context length**: 4000 tokens with smart truncation

**Prompt Engineering**:
```python
system_prompt = """You are a knowledgeable assistant specializing in French geography and culture. 
You provide accurate, informative responses based on the given context.

Key guidelines:
- Use only the information provided in the context
- Be specific and detailed in your responses
- If information is not available in the context, clearly state this
- Cite sections when possible (e.g., "According to the Land section...")
- Maintain a professional, educational tone"""
```

**Context Management**:
- Hierarchical chunk formatting with section headers
- Source attribution and chunk IDs
- Smart truncation with context preservation
- Quality validation and improvement suggestions

### 4. FastAPI Integration 

**Clean Architecture** (`main.py`):
- Pydantic schemas for request/response validation
- Dependency injection for RAG components
- Comprehensive error handling
- CORS support for web interfaces
- Request metrics and monitoring

**Modular Structure**:
```
src/
├── data/           # Scraping, cleaning, chunking
├── embedding.py    # Vector embeddings
├── retrieval/      # Vector store, hybrid search
├── generation.py   # LLM integration, prompts
├── schemas.py      # API schemas
└── main.py         # FastAPI application
```


### Streamlit UI - Interactive Experience

```mermaid
graph LR
    subgraph "🎨 Streamlit Interface"
        A[Welcome Page] --> B[Chat Interface]
        B --> C[Source Explorer]
        C --> D[Analytics Dashboard]
        D --> E[System Status]
        
        B --> B1[Real-time Q&A]
        B --> B2[Example Queries]
        B --> B3[Response History]
        
        C --> C1[Retrieved Chunks]
        C --> C2[Similarity Scores]
        C --> C3[Source Metadata]
        
        D --> D1[Performance Metrics]
        D --> D2[Usage Statistics]
        D --> D3[System Health]
    end
    
    classDef interface fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px
    classDef feature fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    
    class A,B,C,D,E interface
    class B1,B2,B3,C1,C2,C3,D1,D2,D3 feature
```

**Features**:
- **Interactive Chat**: Real-time Q&A with the system
- **Source Explorer**: View retrieved chunks and scores
- **System Analytics**: Performance metrics and statistics
- **French Theme**: Beautiful flag-inspired design
- **Example Queries**: Pre-built geography questions

Access at: `http://localhost:8501`

### Advanced System Features

```mermaid
graph TB
    subgraph "🚀 Production Features"
        A1[Automated Setup]
        A2[Health Monitoring]
        A3[Performance Tracking]
        A4[Error Recovery]
    end
    
    subgraph "🔧 Development Tools"
        B1[Comprehensive Testing]
        B2[API Documentation]
        B3[Debug Logging]
        B4[Configuration Management]
    end
    
    subgraph "📊 Analytics & Insights"
        C1[Request Metrics]
        C2[Response Quality]
        C3[System Statistics]
        C4[Usage Patterns]
    end
    
    A1 --> A2
    A2 --> A3
    A3 --> A4
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    C1 --> C2
    C2 --> C3
    C3 --> C4
    
    classDef production fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    classDef development fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef analytics fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    
    class A1,A2,A3,A4 production
    class B1,B2,B3,B4 development
    class C1,C2,C3,C4 analytics
```

## 📊 Performance & Evaluation

### Sample Outputs

**Query**: "What are the main mountain ranges in France?"

**Retrieved Sources**:
1. **The younger mountains** (score: 0.847)
   - "The Alps and Pyrenees form France's primary mountain barriers..."
2. **Land** (score: 0.732)
   - "France's topography includes ancient massifs and younger mountain chains..."

**Generated Answer**:
> According to the Land section, France's main mountain ranges include the Alps and Pyrenees, which form the country's primary mountain barriers. The Alps section indicates these are part of the younger mountain systems that shape France's southeastern and southwestern borders respectively. The Pyrenees create a natural boundary with Spain, while the Alps extend along the borders with Italy and Switzerland.

### Evaluation Metrics
- **Retrieval Precision**: Relevant chunks in top-K results
- **Answer Faithfulness**: No hallucinations, source-grounded responses
- **Coverage**: Comprehensive use of retrieved context
- **Consistency**: Stable output format and quality
- **Robustness**: Graceful error handling

## 🔧 Configuration

Key settings in `config/config.py`:

```python
# Embedding Configuration
EMBEDDING_TYPE = "tfidf"  # or "sentence-transformers"
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# Generation Configuration
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# Retrieval Configuration
DEFAULT_K = 5
MIN_SCORE = 0.0
MAX_CONTEXT_LENGTH = 4000
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Complete system test
python test_system.py

# Individual component tests
python scripts/data_preprocessing.py --test-mode
python -m pytest tests/ -v
```

Tests cover:
- Data processing pipeline
- Embedding generation
- Retrieval accuracy
- Generation quality
- API endpoint functionality
- Error handling

## 📁 Project Structure

```
france-rag-explorer/
├── config/
│   └── config.py              # Configuration management
├── src/
│   ├── data/
│   │   ├── britannica_scraper.py    # Web scraping
│   │   ├── text_cleaner.py          # Text preprocessing
│   │   ├── document_chunker.py      # Chunking strategies
│   │   └── data_processor.py        # Pipeline orchestration
│   ├── retrieval/
│   │   ├── vector_store.py          # Vector storage
│   │   └── hybrid_retriever.py      # Hybrid search
│   ├── embedding.py           # Embedding models
│   ├── generation.py          # LLM integration
│   └── schemas.py            # API schemas
├── scripts/
│   ├── data_preprocessing.py  # Data pipeline script
│   ├── create_embeddings.py   # Embedding creation
│   ├── run_api.py            # API server
│   └── setup_and_run_app.py  # Complete setup
├── frontend/
│   └── france_rag_ui.py      # Streamlit interface
├── data/                     # Generated data
├── main.py                   # FastAPI application
├── run_app.py               # Main launcher
├── test_system.py           # System tests
├── setup_system.py          # Automated setup
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔍 Design Decisions & Rationale

```mermaid
graph TD
    subgraph "🧩 Chunking Strategy Decision Tree"
        A[Document Input] --> B{Content Type?}
        B -->|Structured| C[Semantic Chunking]
        B -->|Mixed Content| D[Sentence-based]
        B -->|Uniform Text| E[Fixed-length]
        
        E --> E1[✅ 512 tokens<br/>50 overlap]
        E1 --> E2[Rationale:<br/>• Balanced context size<br/>• Prevents info loss<br/>• Fair similarity comparison]
    end
    
    subgraph "🧮 Embedding Model Comparison"
        F[Model Selection] --> G{Priority?}
        G -->|Speed| H[TF-IDF]
        G -->|Quality| I[Sentence Transformers]
        G -->|Balance| J[Hybrid Approach]
        
        H --> H1[✅ Primary Choice<br/>5000 features<br/>1-2 grams]
        H1 --> H2[Benefits:<br/>• Fast inference<br/>• No dependencies<br/>• Good keyword matching]
        
        I --> I1[🔄 Optional<br/>all-MiniLM-L6-v2<br/>384 dimensions]
        I1 --> I2[Benefits:<br/>• Semantic understanding<br/>• Better context<br/>• Higher accuracy]
    end
    
    subgraph "🔍 Retrieval Architecture"
        K[Search Query] --> L[Vector Similarity]
        K --> M[Metadata Filtering]
        L --> N[Combined Scoring]
        M --> N
        N --> O[✅ Top-K Results]
        
        O --> O1[Advantages:<br/>• Semantic + keyword<br/>• Topic filtering<br/>• Configurable precision]
    end
    
    subgraph "🤖 Generation Strategy"
        P[Context + Query] --> Q[Prompt Engineering]
        Q --> R[LLM Processing]
        R --> S[Response Validation]
        
        Q --> Q1[✅ Structured Prompt<br/>System instructions<br/>Context injection]
        Q1 --> Q2[Design:<br/>• Clear guidelines<br/>• Source attribution<br/>• Professional tone]
        
        R --> R1[✅ Llama 3.3 70B<br/>Temperature: 0.3<br/>Max tokens: 1024]
        R1 --> R2[Parameters:<br/>• Balanced creativity<br/>• Comprehensive answers<br/>• Consistent quality]
    end
    
    classDef decision fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef choice fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef rationale fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class A,B,F,G,K,P decision
    class E1,H1,I1,O,Q1,R1 choice
    class E2,H2,I2,O1,Q2,R2 rationale
```

## 📄 License

This project is created for educational purposes as part of an Data Mining course final project.

## 🤝 Contributing

This is an academic project, but feedback and suggestions are welcome!

---

**Built with ❤️ for French Geography Education**

*Explore the diverse landscapes and rich geographical heritage of France through intelligent AI-powered conversations.*
