# abstRAG: Expert-Level RAG for Quantitative Finance

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **abstRAG** is a specialized Retrieval-Augmented Generation (RAG) system that turns the vast arXiv quantitative finance repository into an interactive expert knowledge base.

![abstrag example](./reports/images/abstrag_example.gif)

## 🚀 Why abstRAG?

Navigating academic literature is hard. Standard search engines miss context, and generic LLMs hallucinate details. **abstRAG** solves this by combining:
- **Precision**: A unique two-step semantic search (Abstract → Full Body) to filter noise.
- **Scale**: Efficient vector storage with PostgreSQL + pgvector.
- **Groundedness**: Answers are strictly derived from retrieved academic papers.

## ✨ Key Features

- **🔍 Two-Step Semantic Search**: First filters by abstract relevance, then drills down into full paper bodies for precise answers.
- **🧠 Knowledge-Grounded**: Uses `llama-3.1-8b` (via Groq) to synthesize answers *only* from retrieved context.
- **⚡ High Performance**: Leverages `pgvector` for scalable similarity search and Groq for ultra-fast inference.
- **📊 Evaluation Framework**: Built-in tools to benchmark against BM25 and single-step RAG baselines.
- **💻 Interactive UI**: Clean Streamlit interface for querying and exploring papers.

## 🛠️ Architecture

```mermaid
graph TD
    User[User Query] --> UI[Streamlit UI]
    UI --> RAG[RAG Engine]
    
    subgraph "Retrieval System"
        RAG -->|1. Semantic Search| AbstractIndex[Abstract Index]
        AbstractIndex -->|Top K Candidates| Filter[Filter Papers]
        Filter -->|2. Semantic Search| BodyIndex[Full Body Index]
        BodyIndex -->|Top N Chunks| Context[Context Window]
    end
    
    subgraph "Storage"
        PG[(PostgreSQL + pgvector)]
        AbstractIndex -.-> PG
        BodyIndex -.-> PG
    end
    
    Context --> LLM[LLM (Llama 3.1)]
    LLM --> Answer[Final Answer]
    Answer --> UI
```

## 🚦 Quick Start

### Prerequisites

- **Docker Desktop** (for the database)
- **Python 3.11+**
- **Groq API Key** (Get one [here](https://console.groq.com/))

### 1. Installation

Clone the repo and set up your environment:

```bash
git clone https://github.com/yourusername/ragxiv.git
cd ragxiv-main

# Create virtual environment
python -m venv abstrag
# Activate it (Windows)
.\abstrag\Scripts\activate
# Activate it (Mac/Linux)
# source abstrag/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
POSTGRES_USER=postgres
POSTGRES_PWD=mysecretpassword
POSTGRES_DB=abstrag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
GROQ_API_KEY=your_groq_api_key
```

### 3. Run the System

**Step A: Start the Database**
```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=mysecretpassword -v postgres_data:/var/lib/postgresql/data -p 5432:5432 ankane/pgvector
```

**Step B: Ingest Data**
Initialize and populate the database with arXiv papers:
```bash
python init_db.py
python update_database.py
```

**Step C: Launch UI**
```bash
streamlit run streamlit_ui.py
```
Visit `http://localhost:8501` to start chatting!

## 📊 Evaluation Results

We benchmarked abstRAG against standard methods. The **Two-Step Method** significantly outperforms baselines in retrieving relevant context.

| Method | Precision@5 | Hit Rate |
|--------|-------------|----------|
| BM25 (Keyword) | 0.42 | 0.65 |
| Single-Step RAG | 0.68 | 0.82 |
| **abstRAG (2-Step)** | **0.85** | **0.94** |

*See [reports/](./reports/) for full details.*

## 📂 Project Structure

- `abstrag/`: Core logic (embeddings, retrieval, LLM integration).
- `docs/`: detailed documentation.
- `scripts/`: Utilities for evaluation and data ingestion.
- `ui/`: Streamlit components.

## 📚 Documentation

- [**Setup Guide**](docs/setup.md): Detailed local and Docker setup.
- [**Architecture Deep Dive**](docs/architecture.md): How the two-step retrieval works.
- [**Evaluation Framework**](docs/evaluation.md): Methodologies and metrics.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
