"""Configuration for the different steps of RAG flow"""

import logging
import yaml
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional

# Initialize logger
logger = logging.getLogger(__name__)

# arXiv categories relevant for this project.
# Quantitative finance categories are listed in https://arxiv.org/archive/q-fin
ARXIV_FIELDS = [
    "q-fin.CP",  # Computational Finance
    "q-fin.EC",  # Economics
    "q-fin.GN",  # General Finance
    "q-fin.MF",  # Mathematical Finance
    "q-fin.PM",  # Portfolio Management
    "q-fin.PR",  # Pricing of Securities
    "q-fin.RM",  # Risk Management
    "q-fin.ST",  # Statistical Finance
    "q-fin.TR",  # Trading and Market Microstructure
]


def get_config() -> dict | None:
    config_paths = ["config.yaml", "./config.yaml", "../config.yaml"]

    for path in config_paths:
        config = load_config(path=path)
        if config:
            return config
    return None


class IngestionConfig(BaseModel):
    """Configuration for data ingestion"""
    max_documents_arxiv: int = Field(ge=0, le=10000, description="Maximum documents to fetch from arXiv")
    chunk_size: int = Field(ge=100, le=2000, description="Size of text chunks for embedding")
    chunk_overlap: int = Field(ge=0, le=500, description="Overlap between consecutive chunks")
    chunk_method: Literal["MarkdownTextSplitter"] = Field(description="Method for chunking documents")
    embedding_model_name: str = Field(min_length=1, description="Name of the embedding model")
    deduplication_enabled: bool = Field(default=False, description="Enable chunk deduplication")
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Similarity threshold for deduplication")
    
    @model_validator(mode='after')
    def validate_overlap(self):
        """Validate that chunk_overlap is less than chunk_size"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return self


class RAGConfig(BaseModel):
    """Configuration for RAG pipeline"""
    llm_model: str = Field(min_length=1, description="LLM model name")
    retrieval_method: Literal[
        "pg_semantic_abstract+article",
        "pg_semantic_article",
        "pg_text_article"
    ] = Field(description="Retrieval method to use")
    max_context_tokens: Optional[int] = Field(default=None, ge=1000, le=32000, description="Maximum context tokens")
    max_response_tokens: Optional[int] = Field(default=None, ge=100, le=4000, description="Maximum response tokens")
    cache_ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    abstract_retrieval_k: int = Field(default=10, ge=1, le=100, description="Number of candidate papers to retrieve from abstracts (2-step method)")
    passage_retrieval_k: int = Field(default=15, ge=1, le=200, description="Number of passages to retrieve from candidate papers")
    final_k: int = Field(default=5, ge=1, le=20, description="Number of final passages to return")
    prompt_template_path: Optional[str] = Field(default=None, description="Path to prompt template file")


class APIConfig(BaseModel):
    """Configuration for API settings"""
    timeout_seconds: int = Field(default=60, ge=1, le=300, description="API timeout in seconds")
    rate_limit_per_minute: int = Field(default=60, ge=1, description="General rate limit per minute")
    groq_rate_limit_per_minute: int = Field(default=30, ge=1, description="Groq API rate limit per minute")


class MonitoringConfig(BaseModel):
    """Configuration for monitoring"""
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Log level")
    log_format: Literal["standard", "json"] = Field(default="standard", description="Log format")


class Config(BaseModel):
    """Main configuration model"""
    ingestion: IngestionConfig
    rag: RAGConfig
    api: Optional[APIConfig] = None
    monitoring: Optional[MonitoringConfig] = None


def load_config(path: str) -> dict | None:
    """Load and validate configuration from YAML file
    
    Args:
        path (str): Path to config file
        
    Returns:
        dict | None: Validated configuration dict or None if error
    """
    try:
        with open(path, "r") as file:
            config_dict = yaml.safe_load(file)
            if config_dict:
                # Validate configuration
                config = Config(**config_dict)
                logger.info(f"Successfully loaded and validated configuration from {path}")
                return config.model_dump()
            return None
    except FileNotFoundError:
        logger.debug(f"Config file not found: {path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading config file {path}: {e}")
        return None
