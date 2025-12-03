"""Initialize database"""

import logging
import os
from typing import Final
from dotenv import load_dotenv
from abstrag.core.database import (
    open_db_connection,
    create_embedding_table,
    PostgresParams,
    create_user_feedback_table,
)
from abstrag.config import get_config
from abstrag.core.embedding import get_embedding_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(".env")

config_dict = get_config()
if config_dict:
    from abstrag.config import IngestionConfig
    config_ingestion = IngestionConfig(**config_dict["ingestion"])
else:
    raise ValueError("Failed to load configuration")

EMBEDDING_MODEL_NAME: Final = config_ingestion.embedding_model_name

TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace(
    "-", "_"
)

POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PWD = os.environ["POSTGRES_PWD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]
POSTGRES_HOST = os.environ["POSTGRES_HOST"]
POSTGRES_PORT = os.environ["POSTGRES_PORT"]

# Get embedding model info (using cached model loader)
embedding_transformer = get_embedding_model(EMBEDDING_MODEL_NAME)
word_embedding_dimension = embedding_transformer.get_sentence_embedding_dimension()

postgres_connection_params = PostgresParams(
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    user=POSTGRES_USER,
    pwd=POSTGRES_PWD,
    database=POSTGRES_DB,
)

conn = open_db_connection(connection_params=postgres_connection_params, autocommit=True)

# Create schema
if conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # conn.execute(f"DROP TABLE IF EXISTS {TABLE_EMBEDDING_ARTICLE}")
    # conn.execute(f"DROP TABLE IF EXISTS {TABLE_EMBEDDING_ABSTRACT}")

    # Create tables
    create_embedding_table(
        conn=conn,
        table_name=TABLE_EMBEDDING_ARTICLE,
        embedding_dimension=word_embedding_dimension,
    )
    create_embedding_table(
        conn=conn,
        table_name=TABLE_EMBEDDING_ABSTRACT,
        embedding_dimension=word_embedding_dimension,
    )
    create_user_feedback_table(
        conn=conn,
    )
    logger.info("Database initialized successfully")
else:
    logger.error("Failed to initialize database - connection could not be established")
