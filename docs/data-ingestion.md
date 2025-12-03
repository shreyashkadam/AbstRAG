# Data Ingestion Pipeline

The ingestion pipeline is responsible for fetching academic papers, processing them into machine-readable formats, and indexing them for semantic search.

## Pipeline Overview

The process follows these steps:

1.  **Fetch**: Retrieve papers from arXiv API.
2.  **Parse**: Convert HTML content to clean Markdown.
3.  **Chunk**: Split text into manageable segments.
4.  **Embed**: Generate vector embeddings.
5.  **Store**: Save to PostgreSQL (metadata + vectors).

## 1. Paper Retrieval (arXiv)

We use the `arxiv` Python library to query the arXiv API.
- **Source**: Papers are fetched from the `cond-mat` (Condensed Matter) or `q-fin` (Quantitative Finance) categories.
- **Format**: We prefer fetching the **HTML** version of papers (available for papers post-Dec 2023) rather than PDF parsing, as it preserves structure better.

## 2. Parsing & Cleaning

Raw HTML is noisy. We clean it to maximize retrieval quality:
- **Metadata Removal**: Authors, bibliographies, and appendices are often stripped or handled separately to focus on core content.
- **Markdown Conversion**: We use `markdownify` to convert HTML to Markdown. This preserves headers, lists, and bold text, which helps the chunker understand document structure.
- **Math handling**: LaTeX equations are preserved where possible, though heavy mathematical notation is sometimes simplified for embedding models.

## 3. Chunking Strategy

Effective chunking is critical for RAG.

- **Method**: `MarkdownTextSplitter` (from LangChain).
- **Strategy**: Splits by structure (Headers > Paragraphs > Sentences).
- **Size**: Default chunk size is 500 tokens with 50 token overlap.

*Future improvement*: Semantic chunking (grouping by meaning rather than size) is planned.

## 4. Embedding Generation

We use high-performance models from `sentence-transformers` to convert text into vectors.
- **Model**: `multi-qa-mpnet-base-dot-v1` (Default). Optimized for semantic search.
- **Dimension**: 768 dimensions.

## 5. Storage (PostgreSQL + pgvector)

Data is stored in two main tables:
- **`papers`**: Stores document-level metadata (Title, Abstract, Date, URL).
    - *Abstracts are embedded separately* for the first step of our retrieval process.
- **`chunks`**: Stores individual text chunks with their vector embeddings.
    - Linked to parent papers via Foreign Key.
    - Indexed with HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search.

## Configuration

Ingestion parameters can be tuned in `config.yaml`:

```yaml
ingestion:
  max_documents_arxiv: 50
  chunk_size: 500
  embedding_model_name: "multi-qa-mpnet-base-dot-v1"
```
