"""Chunk documents and obtain embeddings"""

import logging
from tqdm.auto import tqdm
from typing import List, Literal, TypedDict, get_args, TYPE_CHECKING
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from abstrag.utils import normalize_vector

# Initialize logger
logger = logging.getLogger(__name__)

# Global cache for embedding models
_embedding_model_cache = {}


# Default embedding parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_METHOD = "MarkdownTextSplitter"
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"

ChunkMethod = Literal["MarkdownTextSplitter"]
SentenceTransformerModels = Literal["multi-qa-mpnet-base-dot-v1",]
EmbeddingModel = Literal[SentenceTransformerModels]


class ChunkParams(TypedDict):
    method: ChunkMethod
    size: int
    overlap: int


class Embedding(TypedDict):
    model: EmbeddingModel
    dimension: int
    content: List[str]
    embedding: np.ndarray


class PaperEmbedding(TypedDict):
    id: str
    content: str
    embeddings: np.ndarray


def chunk_document(document: str, chunk_params: ChunkParams) -> List[str]:
    """Split a given document using the selected method

    Args:
        document (str): Markdown text of a document
        chunk_params (ChunkParams): Specification of the chunking
            method that will be used

    Raises:
        ValueError: The selected chunk method has not been
            implemented yet

    Returns:
        List[str]: List of strings containing the different
            document chunks
    """
    if chunk_params["method"] == "MarkdownTextSplitter":
        chunks = chunk_markdown_recursive(document=document, chunk_params=chunk_params)
    else:
        raise ValueError(f"ChunkMethod {chunk_params['method']} not implemented")
    return chunks


def chunk_markdown_recursive(document: str, chunk_params: ChunkParams) -> List[str]:
    """Recursive Character Text Splitter using Markdown separators

    Args:
        document (str): Markdown text of a document
        chunk_params (ChunkParams): Specification of the chunking
            method that will be used
    Returns:
        List[str]: List of strings containing the different
            document chunks
    """
    chunk_size = chunk_params["size"]
    chunk_overlap = chunk_params["overlap"]

    # Initialize splitter with markdown-aware separators
    separators = ["\n\n## ", "\n\n### ", "\n\n#### ", "\n\n##### ", "\n\n###### ", "\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=separators
    )

    chunks = splitter.create_documents([document])

    # To list of strings
    chunks = [chunk.page_content for chunk in chunks]

    return chunks


def document_embedding(chunks: List[str], embedding_model: EmbeddingModel) -> Embedding:
    """Create a vector embedding for a list of document chunks

    Args:
        chunks (List[str]): List of strings containing the different
            document chunks
        embedding_model (EmbeddingModel): Name of the embedding model

    Raises:
        ValueError: The selected embedding method has not been
            implemented yet

    Returns:
        Embedding: Normalized vector embedding as an np.array together with
            the original chunks and model details

    """
    if embedding_model in get_args(SentenceTransformerModels):
        embedding = document_embedding_sentence_transformers(
            chunks=chunks, embedding_model=embedding_model
        )
    else:
        raise ValueError(f"EmbeddingModel {embedding_model} not implemented")
    return embedding


def get_embedding_model(model_name: EmbeddingModel) -> "SentenceTransformer":
    """Get or load embedding model with caching
    
    Args:
        model_name (EmbeddingModel): Name of the embedding model
        
    Returns:
        SentenceTransformer: Loaded embedding model (cached)
    """
    # Import here to avoid loading PyTorch at module import time
    from sentence_transformers import SentenceTransformer
    
    if model_name not in _embedding_model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        try:
            _embedding_model_cache[model_name] = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            raise ValueError(f"Unable to load embedding model {model_name}") from e
    else:
        logger.debug(f"Using cached embedding model: {model_name}")
    
    return _embedding_model_cache[model_name]


def document_embedding_sentence_transformers(
    chunks: List[str], embedding_model: EmbeddingModel
) -> Embedding:
    """Sentence Transformer embedding with batch processing

    Args:
        chunks (List[str]): List of strings containing the different
            document chunks
        embedding_model (EmbeddingModel): Name of the embedding model

    Returns:
        Embedding: Normalized vector embedding as an np.array together with
            the original chunks and model details
    """
    # Get cached embedding model
    embedding_transformer = get_embedding_model(embedding_model)
    word_embedding_dimension = embedding_transformer.get_sentence_embedding_dimension()

    # Batch encode all chunks at once for better performance
    if len(chunks) == 0:
        logger.warning("No chunks provided for embedding")
        document_embeddings = np.empty(shape=(0, word_embedding_dimension))
    else:
        logger.info(f"Encoding {len(chunks)} chunks in batch")
        # Use batch encoding with progress bar
        # normalize_embeddings=True already normalizes, so we don't need to normalize again
        document_embeddings = embedding_transformer.encode(
            chunks,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        # Ensure it's a numpy array
        document_embeddings = np.array(document_embeddings)

    embedding = Embedding(
        model=embedding_transformer,
        dimension=word_embedding_dimension,
        content=chunks,
        embedding=document_embeddings,
    )
    return embedding

