# Setup Guide

This guide covers the complete setup process for running **abstRAG** locally.

## Prerequisites

1.  **Python 3.11+**
2.  **Docker Desktop** (for PostgreSQL + pgvector)
3.  **Groq API Key** ([Get it here](https://console.groq.com/))

## 1. Environment Setup

Clone the repository and set up your Python environment:

```bash
# Create virtual environment
python -m venv abstrag

# Activate environment
# Windows:
.\abstrag\Scripts\activate
# Mac/Linux:
# source abstrag/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
# Note: Windows users might need the extra index for CUDA support:
# pip install --extra-index-url https://pypi.nvidia.com -r requirements.txt
```

## 2. Configuration

Create a `.env` file in the project root:

```env
POSTGRES_USER=postgres
POSTGRES_PWD=mysecretpassword
POSTGRES_DB=abstrag_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
GROQ_API_KEY=your_groq_api_key
```

Modify `config.yaml` if you need to adjust ingestion parameters (e.g., number of papers, chunk size).

## 3. Database Setup

Run PostgreSQL with `pgvector` using Docker:

```bash
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -v postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  ankane/pgvector
```

Initialize the database schema:

```bash
python init_db.py
```

## 4. Data Ingestion

Fetch papers from arXiv and populate the vector database:

```bash
python update_database.py
```
*Note: This process fetches papers, converts them to markdown, chunks them, and generates embeddings. It may take a few minutes depending on `max_documents_arxiv` setting.*

## 5. Running the Application

Launch the Streamlit interface:

```bash
streamlit run streamlit_ui.py
```
Access the UI at `http://localhost:8501`.

### Optional: Monitoring Dashboard

To view feedback metrics:
```bash
streamlit run streamlit_feedback_monitor.py --server.port 8500
```

## Troubleshooting

- **Database Connection Failed**: Ensure the Docker container is running (`docker ps`) and credentials in `.env` match the `docker run` command.
- **CUDA/Torch Issues**: If on Windows without a GPU, ensure you installed the CPU versions of torch if the default install fails.
- **Groq API Errors**: Verify your API key and check for rate limits.
