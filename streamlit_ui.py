"""Modern Streamlit frontend for abstRAG"""

import logging
import os
import uuid
from dotenv import load_dotenv
from typing import List, Final, Generator, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st

import psycopg
from abstrag.core.database import (
    open_db_connection,
    PostgresParams,
    SemanticSearch,
    UserFeedback,
    insert_user_feedback,
    check_connection_health,
)
from abstrag.core.retrieval import retrieve_similar_documents
from abstrag.core.llm import GroqParams, build_rag_prompt
from abstrag.config import get_config, Config
from abstrag.utils.cache import QueryCache
from abstrag.utils.rate_limiter import RateLimiter
from abstrag.utils.metrics import get_metrics_collector
from abstrag.utils.logging_config import setup_logging
from groq import Groq

# Import UI components
from ui.components.header import render_header
from ui.components.sidebar import render_sidebar
from ui.components.metrics_display import render_metrics
from ui.components.references_display import render_references

load_dotenv(".env")

# Configure logging (will be reconfigured based on config)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

config_dict = get_config()
if config_dict:
    # Convert dict to Config object for type safety
    from abstrag.config import Config, IngestionConfig, RAGConfig, APIConfig, MonitoringConfig
    config = Config(
        ingestion=IngestionConfig(**config_dict["ingestion"]),
        rag=RAGConfig(**config_dict["rag"]),
        api=APIConfig(**config_dict.get("api", {})) if config_dict.get("api") else None,
        monitoring=MonitoringConfig(**config_dict.get("monitoring", {})) if config_dict.get("monitoring") else None,
    )
    config_ingestion = config.ingestion
    config_rag = config.rag
else:
    st.error("Failed to load configuration")
    st.stop()

# Setup logging based on config
log_level = config.monitoring.log_level if config.monitoring else "INFO"
log_format = config.monitoring.log_format if config.monitoring else "standard"
setup_logging(level=log_level, format_type=log_format, json_output=(log_format == "json"))
logger = logging.getLogger(__name__)

# Page configuration with modern settings
st.set_page_config(
    page_icon="📚",
    layout="wide",
    page_title="abstRAG - RAG with arXiv",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/abstrag',
        'Report a bug': 'https://github.com/your-repo/abstrag/issues',
        'About': "# abstRAG\nRetrieval-Augmented Generation with arXiv Knowledge Base"
    }
)

# Load custom CSS
try:
    with open("ui/styles/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass  # CSS file is optional

# Set variables
EMBEDDING_MODEL_NAME: Final = config_ingestion.embedding_model_name
TABLE_EMBEDDING_ARTICLE = f"embedding_article_{EMBEDDING_MODEL_NAME}".replace("-", "_")
TABLE_EMBEDDING_ABSTRACT = f"embedding_abstract_{EMBEDDING_MODEL_NAME}".replace("-", "_")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

LLM_MODEL: Final = "groq"
LLM_MODEL_PARAMS = GroqParams(api_key=GROQ_API_KEY, model=config_rag.llm_model)
RETRIEVAL_METHOD: Final = config_rag.retrieval_method

# Initialize utilities
query_cache = QueryCache(ttl_seconds=config_rag.cache_ttl_seconds)
rate_limiter = RateLimiter(requests_per_minute=config.api.groq_rate_limit_per_minute if config.api else 30)
metrics_collector = get_metrics_collector() if (config.monitoring and config.monitoring.metrics_enabled) else None

postgres_connection_params = PostgresParams(
    host=os.environ["POSTGRES_HOST"],
    port=os.environ["POSTGRES_PORT"],
    user=os.environ["POSTGRES_USER"],
    pwd=os.environ["POSTGRES_PWD"],
    database=os.environ["POSTGRES_DB"],
)

# Initialize Groq client with timeout from config
groq_timeout = config.api.timeout_seconds if config.api else 60
client = Groq(api_key=GROQ_API_KEY, timeout=groq_timeout)
logger.info(f"Initialized Groq client with timeout: {groq_timeout}s")


@st.cache_resource
def open_connection():
    conn = open_db_connection(
        connection_params=postgres_connection_params, autocommit=True
    )
    return conn


@st.cache_resource
def create_unique_id() -> str:
    unique_id = str(uuid.uuid4())
    return unique_id


unique_id = create_unique_id()
conn = open_connection()

# Check connection health
if conn and not check_connection_health(conn):
    st.error("⚠️ Database connection is unhealthy. Please refresh the page.")
    logger.warning("Database connection health check failed")
    st.stop()

# Render header
render_header()

# Render sidebar
render_sidebar(metrics_collector=metrics_collector)


def validate_query(query: str) -> Tuple[bool, str]:
    """Validate user query"""
    if not query or len(query.strip()) < 3:
        return False, "Query is too short. Please enter at least 3 characters."
    if len(query) > 1000:
        return False, "Query is too long. Please keep it under 1000 characters."
    return True, ""


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "question_state" not in st.session_state:
    st.session_state.question_state = False

if "feedback_value" not in st.session_state:
    st.session_state.feedback_value = None

if "user_feedback" not in st.session_state:
    st.session_state.user_feedback = None


def fbcb():
    """Feedback callback function"""
    if st.session_state.feedback_value is not None:
        logger.info(f"User feedback: {st.session_state.feedback_value}")


# Display welcome message if no chat history
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome to abstRAG! Ask me anything about quantitative finance research papers from arXiv.")

# Display chat messages from history
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def build_user_feedback(
    user_question: str,
    system_answer: str,
    references: Optional[str] = None,
    satisfied: Optional[int] = None,
    elapsed_time: Optional[timedelta] = None,
    feedback_timestamp: Optional[datetime] = datetime.now(),
) -> UserFeedback:
    """Build user feedback object"""
    return UserFeedback(
        user_id=unique_id,
        question=user_question,
        answer=system_answer,
        thumbs=satisfied,
        documents_retrieved=references,
        similarity=None,
        relevance=None,
        llm_model=None,
        embedding_model=None,
        elapsed_time=elapsed_time,
        feedback_timestamp=feedback_timestamp,
    )


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    chunk_count = 0
    content_chunks = 0
    logger.info("Starting to iterate over chat completion stream")
    try:
        import time
        last_chunk_time = time.time()
        timeout_seconds = 30  # 30 second timeout between chunks (more aggressive)
        
        for chunk in chat_completion:
            current_time = time.time()
            time_since_last = current_time - last_chunk_time
            
            # Check for timeout between chunks
            if time_since_last > timeout_seconds and chunk_count > 0:
                logger.error(f"Stream timeout: No chunks received for {timeout_seconds} seconds (last chunk was #{chunk_count})")
                raise TimeoutError(f"Stream timeout: No chunks received for {timeout_seconds} seconds")
            
            chunk_count += 1
            last_chunk_time = current_time
            
            if chunk_count == 1:
                logger.info(f"Received first chunk from Groq API stream")
            elif chunk_count % 10 == 0:
                logger.info(f"Received {chunk_count} chunks so far ({content_chunks} with content)")
            
            if chunk.choices and len(chunk.choices) > 0:
                # Check if this is a finish reason (stream ending)
                if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                    logger.info(f"Stream finished with reason: {chunk.choices[0].finish_reason} (total chunks: {chunk_count})")
                    break
                
                if chunk.choices[0].delta.content:
                    content_chunks += 1
                    if content_chunks <= 3 or content_chunks % 20 == 0:
                        logger.debug(f"Yielding chunk {chunk_count} (content chunk #{content_chunks}, length: {len(chunk.choices[0].delta.content)})")
                    yield chunk.choices[0].delta.content
                elif chunk_count <= 5:  # Log first few chunks even if empty
                    logger.debug(f"Chunk {chunk_count} has no content (delta: {chunk.choices[0].delta})")
            else:
                logger.warning(f"Chunk {chunk_count} has no choices")
                
        logger.info(f"Finished streaming response. Total chunks: {chunk_count}, Content chunks: {content_chunks}")
    except TimeoutError:
        logger.error(f"Streaming timeout occurred after {chunk_count} chunks ({content_chunks} with content)")
        raise
    except StopIteration:
        logger.info(f"Stream ended normally (StopIteration) after {chunk_count} chunks ({content_chunks} with content)")
        raise
    except Exception as e:
        logger.error(f"Error in generate_chat_responses after {chunk_count} chunks: {e}", exc_info=True)
        raise


# Chat input with better styling
user_query = st.chat_input(
    "Ask a question about quantitative finance research...",
    key="chat_input"
)

if user_query:
    # Validate query
    is_valid, error_msg = validate_query(user_query)
    if not is_valid:
        st.warning(f"⚠️ {error_msg}")
        logger.warning(f"Invalid query rejected: {error_msg}")
    else:
        st.session_state.question_state = True


if st.session_state.question_state:
    st.session_state.messages.append({"role": "user", "content": user_query})
    ini_time = datetime.now()

    with st.chat_message("user", avatar="👨‍💻"):
        st.markdown(user_query)

    # Fetch response with modern progress indicators
    full_response = None
    relevant_documents = None
    
    try:
        # Use a more modern status display
        with st.status("🔍 **Searching academic papers...**", expanded=True) as status:
            # Step 1: Search abstracts (2-step pipeline handles this internally)
            status.update(label="📑 Searching paper abstracts...", state="running")
            semantic_search_abstract = SemanticSearch(
                query=user_query,
                table=TABLE_EMBEDDING_ABSTRACT,
                similarity_metric="<#>",
                embedding_model=EMBEDDING_MODEL_NAME,
                max_documents=config_rag.abstract_retrieval_k,
            )

            # Step 2: Search article content (2-step pipeline handles this internally)
            status.update(label="📄 Searching article content...", state="running")
            semantic_search_article = SemanticSearch(
                query=user_query,
                table=TABLE_EMBEDDING_ARTICLE,
                similarity_metric="<#>",
                embedding_model=EMBEDDING_MODEL_NAME,
                max_documents=config_rag.passage_retrieval_k,
            )

            semantic_search_hierarchy = [
                semantic_search_abstract,
                semantic_search_article,
            ]

            # Retrieve documents with 2-step pipeline
            status.update(label="🔎 Retrieving relevant documents...", state="running")
            relevant_documents = retrieve_similar_documents(
                conn=conn,
                retrieval_method=RETRIEVAL_METHOD,
                retrieval_parameters=semantic_search_hierarchy,
                abstract_k=config_rag.abstract_retrieval_k,
                passage_k=config_rag.passage_retrieval_k,
                final_k=config_rag.final_k,
            )
            
            logger.info(f"Retrieved {len(relevant_documents['documents'])} documents from {len(set(relevant_documents['references']))} papers")
            
            # Step 4: Build prompt with context window management
            status.update(label="📝 Preparing response with context management...", state="running")
            prompt = build_rag_prompt(
                user_question=relevant_documents["question"],
                context=relevant_documents["documents"],
                model=config_rag.llm_model,
                max_context_tokens=config_rag.max_context_tokens,
            )

            # Step 6: Apply rate limiting
            rate_limiter.wait_if_needed(key="groq", tokens=1)
            
            # Step 7: Generate answer
            status.update(label="🤖 Generating answer with LLM...", state="running")
            if metrics_collector:
                metrics_collector.record_api_call("groq")
            
            logger.info(f"Calling Groq API with model: {config_rag.llm_model}, prompt length: {len(prompt)} chars")
            logger.debug(f"Prompt preview: {prompt[:200]}...")
            try:
                import time
                start_time = time.time()
                chat_completion = client.chat.completions.create(
                    model=config_rag.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                elapsed = time.time() - start_time
                logger.info(f"Successfully initiated Groq API streaming response (took {elapsed:.2f}s)")
            except Exception as api_error:
                logger.error(f"Error calling Groq API: {api_error}", exc_info=True)
                st.error(f"🚨 **API Error**\n\nFailed to connect to Groq API: {str(api_error)}\n\nPlease check your API key and network connection.", icon="🚨")
                raise

            # Generate response
            logger.info("Starting to stream response from Groq API")
            full_response = None
            try:
                import time
                import threading
                stream_start = time.time()
                
                # Create a wrapper generator with timeout protection
                def safe_generator(base_generator):
                    """Wrapper generator that handles timeouts and errors"""
                    try:
                        for item in base_generator:
                            yield item
                    except Exception as gen_error:
                        logger.error(f"Error in generator: {gen_error}", exc_info=True)
                        raise
                
                # Initialize full_response before the try block
                full_response = None
                response_parts = []
                
                with st.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    safe_gen = safe_generator(chat_responses_generator)
                    
                    # Use manual streaming with error recovery
                    try:
                        for chunk in safe_gen:
                            response_parts.append(chunk)
                        full_response = "".join(response_parts)
                        st.markdown(full_response)
                        logger.info(f"Streamed response successfully ({len(full_response)} chars)")
                    except Exception as stream_error:
                        # If streaming fails, show what we got so far
                        partial_response = "".join(response_parts)
                        if partial_response:
                            st.markdown(partial_response)
                            st.warning("⚠️ Stream interrupted. Partial response shown above.")
                            full_response = partial_response
                            logger.warning(f"Stream interrupted but partial response saved ({len(partial_response)} chars)")
                        else:
                            logger.error(f"Stream failed with no partial response: {stream_error}")
                            raise stream_error
                
                stream_elapsed = time.time() - stream_start
                logger.info(f"Successfully received and displayed response from Groq API (streaming took {stream_elapsed:.2f}s)")
                logger.info(f"full_response type: {type(full_response)}, length: {len(full_response) if full_response else 0}")
                
            except StopIteration:
                logger.warning("Stream ended unexpectedly (StopIteration)")
                if full_response:
                    st.info("⚠️ Stream ended early, but partial response was received.")
                else:
                    st.warning("⚠️ The response stream ended unexpectedly. Please try again.")
            except TimeoutError as timeout_err:
                logger.error(f"Streaming timeout: {timeout_err}")
                st.error(f"⏱️ **Timeout Error**\n\nThe response took too long to generate.\n\n{str(timeout_err)}\n\nPlease try a shorter query or try again.", icon="⏱️")
                # Try non-streaming fallback
                logger.info("Attempting non-streaming fallback after timeout")
                try:
                    non_stream_response = client.chat.completions.create(
                        model=config_rag.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=False,
                    )
                    full_response = non_stream_response.choices[0].message.content
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(full_response)
                    logger.info("Non-streaming fallback succeeded after timeout")
                except Exception as fallback_error:
                    logger.error(f"Non-streaming fallback also failed: {fallback_error}", exc_info=True)
            except Exception as stream_error:
                logger.error(f"Error streaming response from Groq API: {stream_error}", exc_info=True)
                st.error(f"🚨 **Streaming Error**\n\nFailed to receive response: {str(stream_error)}\n\nPlease try again.", icon="🚨")
                # Try non-streaming fallback
                logger.info("Attempting non-streaming fallback")
                try:
                    non_stream_response = client.chat.completions.create(
                        model=config_rag.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=False,
                    )
                    full_response = non_stream_response.choices[0].message.content
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(full_response)
                    logger.info("Non-streaming fallback succeeded")
                except Exception as fallback_error:
                    logger.error(f"Non-streaming fallback also failed: {fallback_error}", exc_info=True)
                    raise stream_error  # Re-raise original error
            
            status.update(label="✅ Complete!", state="complete")
                
    except psycopg.DatabaseError as e:
        logger.error(f"Database error: {e}", exc_info=True)
        st.error(f"🗄️ **Database Error**\n\n{str(e)}\n\nPlease try again.", icon="🗄️")
    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        st.error(f"⚠️ **Validation Error**\n\n{str(e)}", icon="⚠️")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        st.error(f"🚨 **Unexpected Error**\n\n{str(e)}\n\nPlease try again.", icon="🚨")

    # Process response and display results
    logger.info(f"Processing response: full_response={'present' if full_response else 'None'}, relevant_documents={'present' if relevant_documents else 'None'}")
    if full_response and relevant_documents:
        end_time = datetime.now()
        response_time = (end_time - ini_time).total_seconds()
        
        try:
            # Record metrics
            logger.info("Recording metrics")
            if metrics_collector:
                metrics_collector.record_query_latency(response_time)
                metrics_collector.record_retrieval_quality(
                    num_documents=len(relevant_documents["documents"]),
                    num_papers=len(set(relevant_documents["references"])),
                )
            
            # Store response
            logger.info("Storing response in session state")
            if isinstance(full_response, str):
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                combined_response = "\n".join(str(item) for item in full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": combined_response}
                )

            # Display references in a modern way
            logger.info(f"Displaying {len(relevant_documents['references'])} references")
            with st.chat_message("assistant", avatar="📚"):
                st.markdown("### 📚 Referenced Papers")
                for i, ref in enumerate(relevant_documents["references"], 1):
                    st.markdown(f"{i}. [{ref}]({ref})")
            
            st.session_state.messages.append(
                {"role": "assistant", "content": f"📚 Referenced Papers:\n" + "\n".join([f"{i}. {ref}" for i, ref in enumerate(relevant_documents['references'], 1)])}
            )
            logger.info("Successfully processed response and references")
        except Exception as process_error:
            logger.error(f"Error processing response: {process_error}", exc_info=True)
            st.error(f"⚠️ Error displaying results: {str(process_error)}")
            raise

        # Display metrics in an expandable section
        with st.expander("📊 **Retrieval Details & Performance**", expanded=False):
            avg_latency = metrics_collector.get_average_latency() if metrics_collector else None
            render_metrics(
                num_documents=len(relevant_documents["documents"]),
                num_papers=len(set(relevant_documents["references"])),
                response_time=response_time,
                avg_latency=avg_latency,
            )
            
            # Additional info
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model:** {config_rag.llm_model}")
                st.write(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
            with col2:
                st.write(f"**Retrieval Method:** {RETRIEVAL_METHOD}")

        # Feedback section with standard buttons to avoid potential widget crashes
        st.markdown("---")
        logger.info("Rendering feedback section")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**Was this response helpful?**")
        
        with col2:
            # Using columns for buttons to simulate thumbs up/down
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("👍", key="thumbs_up"):
                    st.session_state.feedback_value = 1
                    fbcb()
            with btn_col2:
                if st.button("👎", key="thumbs_down"):
                    st.session_state.feedback_value = 0
                    fbcb()

        with col3:
            if st.session_state.feedback_value is not None:
                st.success("Thank you for your feedback! 👍" if st.session_state.feedback_value == 1 else "We'll work to improve! 👎")

        logger.info(f"Feedback rendered. feedback_value={st.session_state.feedback_value}")

        # Capture feedback details
        user_feedback = build_user_feedback(
            user_question=user_query,
            system_answer=full_response if isinstance(full_response, str) else str(full_response),
            references=";".join(relevant_documents["references"]),
            satisfied=st.session_state.feedback_value,
            elapsed_time=end_time - ini_time,
        )
        st.session_state.user_feedback = user_feedback
        logger.info("User feedback object created and stored in session state")

    st.session_state.question_state = False
    logger.info("Reset question_state to False")


# Handle feedback submission
if st.session_state.feedback_value is not None and st.session_state.user_feedback:
    logger.info(f"Processing feedback submission. Value: {st.session_state.feedback_value}")
    try:
        st.session_state.user_feedback["thumbs"] = st.session_state.feedback_value
        insert_user_feedback(conn=conn, feedback=st.session_state.user_feedback)
        logger.info(f"User feedback stored: {st.session_state.user_feedback['thumbs']}")
        st.session_state.feedback.append(st.session_state.user_feedback)
    except Exception as e:
        logger.error(f"Error storing user feedback: {e}", exc_info=True)
        st.warning("⚠️ Could not save feedback. Please try again.")

logger.info("Script execution completed")
