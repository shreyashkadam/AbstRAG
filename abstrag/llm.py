"""Define LLM functionality by connecting to an external API"""

import logging
import os
from typing import Literal, TypedDict, Union, List, Optional
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from abstrag.utils.token_counter import count_tokens, get_model_token_limit, truncate_context

# Initialize logger
logger = logging.getLogger(__name__)

GroqModels = Literal[
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "llama3-groq-70b-8192-tool-use-preview",
    "llama-3.3-70b-versatile",
]


class GroqParams(TypedDict):
    api_key: str
    model: GroqModels


LLM = Literal["groq"]
LLMParameters = Union[GroqParams]


class LLMResponse(TypedDict):
    response: str
    model: str


def llm_chat_completion(
    query: str, llm_model: LLM, llm_parameters: LLMParameters
) -> LLMResponse:
    if llm_model == "groq":
        llm_response = groq_chat_completion(query=query, llm_parameters=llm_parameters)
    else:
        raise ValueError(f"LLM model {llm_model} not implemented")
    return llm_response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def groq_chat_completion(query: str, llm_parameters: GroqParams) -> LLMResponse:
    """Generate chat completion with retry logic
    
    Args:
        query (str): User query/prompt
        llm_parameters (GroqParams): Groq API parameters
        
    Returns:
        LLMResponse: LLM response with answer and model info
        
    Raises:
        Exception: If API call fails after retries
    """
    try:
        client = Groq(api_key=llm_parameters["api_key"])

        logger.debug(f"Calling Groq API with model: {llm_parameters['model']}")
        response = client.chat.completions.create(
            model=llm_parameters["model"],
            messages=[{"role": "user", "content": query}],
        )

        llm_response = LLMResponse(
            response=response.choices[0].message.content,
            model=f"groq  -  {llm_parameters['model']}",
        )
        logger.debug(f"Successfully received response from Groq API")
        return llm_response
    except Exception as e:
        logger.error(f"Error calling Groq API: {e}")
        raise


def build_rag_prompt(
    user_question: str,
    context: List[str],
    model: Optional[str] = None,
    max_context_tokens: Optional[int] = None,
) -> str:
    """Build RAG prompt with context window management
    
    Args:
        user_question: User's question
        context: List of context documents
        model: Model name for token limit calculation
        max_context_tokens: Maximum tokens for context (overrides model default)
        
    Returns:
        Formatted prompt string
    """
    # Determine token limit
    if max_context_tokens is None and model:
        max_context_tokens = get_model_token_limit(model)
        # Reserve tokens for prompt template and response
        max_context_tokens = max_context_tokens - 1000  # Reserve for template + response
    
    # Truncate context if needed
    if max_context_tokens:
        context = truncate_context(
            context_chunks=context,
            max_tokens=max_context_tokens,
            query=user_question,
            reserve_tokens=500,
        )
    
    document_string = " \n\n ".join([f"{ document} " for document in context])
    
    # Count tokens for logging
    prompt_text = f"""
    You are an expert in quantitative finance. Answer QUESTION but limit your information to what is inside CONTEXT.

    QUESTION: {user_question}

    CONTEXT: {document_string}
    """
    
    if model:
        token_count = count_tokens(prompt_text, model)
        logger.info(f"Built prompt with {token_count} tokens for model {model}")
    
    return prompt_text


def build_retrieval_evaluation_prompt(document: str, number_questions: int) -> str:
    prompt = f"""
    You are an expert in quantitative finance. Formulate {number_questions}
    questions that you would ask based on an academic paper document. The
    questions should be complete and not too short. If possible, use as
    fewer words as possible from the document.

    The document: {document}


    Provide the output in parsable JSON without using code blocks:
    ["question 1", "question 2", ..., ]
    """
    return prompt

