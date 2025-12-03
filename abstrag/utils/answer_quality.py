"""Answer quality evaluation metrics"""

import logging
import json
import re
import time
from typing import Dict, Optional, List
from groq import Groq

try:
    from abstrag.utils.token_counter import count_tokens
except ImportError:
    # Fallback if token_counter is not available
    def count_tokens(text: str, model: Optional[str] = None) -> int:
        # Approximate: 4 characters per token
        return len(text) // 4

logger = logging.getLogger(__name__)

# Threshold for using concise prompts proactively (in tokens)
LARGE_PROMPT_THRESHOLD = 4000  # If input prompt is > 4000 tokens, use concise prompt (lowered to prevent truncation)


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON object from text using multiple strategies
    
    Args:
        text: Text that may contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
    
    # Strategy 1: Try direct JSON parsing
    text_stripped = text.strip()
    if text_stripped.startswith('{') and text_stripped.endswith('}'):
        return text_stripped
    
    # Strategy 2: Extract from markdown code blocks (handle nested JSON)
    # Find code blocks and extract JSON using brace matching
    code_block_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in code_block_patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            potential_json = match.group(1).strip()
            # Check if it's a valid JSON object by counting braces
            if potential_json.startswith('{') and potential_json.count('{') == potential_json.count('}'):
                return potential_json
    
    # Strategy 3: Find JSON object using regex (find first { ... } pair)
    brace_count = 0
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return text[start_idx:i+1].strip()
    
    # Strategy 4: Try to find JSON-like structure and clean it
    # Look for patterns like "Relevance": "..." or "FactualAccuracy": "..."
    if '"Relevance"' in text or '"FactualAccuracy"' in text:
        # Try to extract the JSON object
        start = text.find('{')
        if start != -1:
            # If we found a start, try to find the end
            end = text.rfind('}')
            if end != -1 and end > start:
                potential_json = text[start:end+1]
                return potential_json.strip()
            else:
                # JSON might be truncated - extract from start to end of text
                potential_json = text[start:].strip()
                return potential_json
    
    return None


def clean_json_string(json_str: str) -> str:
    """Clean JSON string to fix common issues
    
    Args:
        json_str: JSON string that may have issues
        
    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix single quotes to double quotes (common mistake)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    
    # Remove control characters that might break JSON
    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
    
    return json_str.strip()


def repair_truncated_json(json_str: str) -> str:
    """Attempt to repair truncated JSON by closing strings and objects
    
    Args:
        json_str: Potentially truncated JSON string
        
    Returns:
        Repaired JSON string
    """
    if not json_str:
        return json_str
    
    # Remove trailing whitespace but preserve structure
    json_str = json_str.rstrip()
    
    # Count braces and brackets to see if JSON is incomplete
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Check if we're in the middle of a string at the end
    in_string = False
    escape_next = False
    
    # Traverse backwards to find if we're inside a string
    for i in range(len(json_str) - 1, -1, -1):
        char = json_str[i]
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            break
    
    # If we're in a string, close it first
    if in_string:
        json_str = json_str + '"'
    
    # Close any unclosed arrays first (they're nested inside objects)
    for _ in range(open_brackets - close_brackets):
        json_str = json_str.rstrip().rstrip(',') + ']'
    
    # Close any unclosed objects
    for _ in range(open_braces - close_braces):
        json_str = json_str.rstrip().rstrip(',') + '}'
    
    return json_str


def extract_partial_json_fields(json_str: str, response_type: str) -> Optional[Dict]:
    """Extract partial JSON fields using regex when JSON is too broken to parse
    
    Args:
        json_str: Broken JSON string
        response_type: Type of response ("relevance" or "factual_accuracy")
        
    Returns:
        Dictionary with extracted fields or None
    """
    result = {}
    
    if response_type == "relevance":
        # Try to extract Relevance field
        relevance_match = re.search(r'"Relevance"\s*:\s*"([^"]+)"', json_str)
        if relevance_match:
            result["Relevance"] = relevance_match.group(1)
        
        # Try to extract Explanation (may be truncated)
        explanation_match = re.search(r'"Explanation"\s*:\s*"([^"]*)', json_str)
        if explanation_match:
            result["Explanation"] = explanation_match.group(1)
    
    elif response_type == "factual_accuracy":
        # Try to extract FactualAccuracy field
        accuracy_match = re.search(r'"FactualAccuracy"\s*:\s*"([^"]+)"', json_str)
        if accuracy_match:
            result["FactualAccuracy"] = accuracy_match.group(1)
        
        # Try to extract Explanation (may be truncated)
        explanation_match = re.search(r'"Explanation"\s*:\s*"([^"]*)', json_str)
        if explanation_match:
            result["Explanation"] = explanation_match.group(1)
        
        # Try to extract SupportedClaims (partial - handles truncated arrays)
        supported_match = re.search(r'"SupportedClaims"\s*:\s*\[(.*)', json_str, re.DOTALL)
        if supported_match:
            claims_str = supported_match.group(1)
            # Extract individual claim strings (handles truncated strings)
            claims = re.findall(r'"([^"]*)"', claims_str)
            # Also try to extract the last incomplete string if present
            incomplete_match = re.search(r'"([^"]*?)$', claims_str)
            if incomplete_match and incomplete_match.group(1):
                claims.append(incomplete_match.group(1))
            if claims:
                result["SupportedClaims"] = claims
        
        # Try to extract UnsupportedClaims (partial - handles truncated arrays)
        unsupported_match = re.search(r'"UnsupportedClaims"\s*:\s*\[(.*)', json_str, re.DOTALL)
        if unsupported_match:
            claims_str = unsupported_match.group(1)
            claims = re.findall(r'"([^"]*)"', claims_str)
            # Also try to extract the last incomplete string if present
            incomplete_match = re.search(r'"([^"]*?)$', claims_str)
            if incomplete_match and incomplete_match.group(1):
                claims.append(incomplete_match.group(1))
            if claims:
                result["UnsupportedClaims"] = claims
    
    return result if result else None


def parse_judge_response(judge_response: str, response_type: str = "relevance", is_truncated: bool = False) -> Optional[Dict]:
    """Parse judge response with robust JSON extraction
    
    Args:
        judge_response: Raw response from LLM judge
        response_type: Type of response ("relevance" or "factual_accuracy")
        is_truncated: Whether the response was truncated (for repair attempts)
        
    Returns:
        Parsed dictionary or None if parsing fails
    """
    if not judge_response:
        return None
    
    # Extract JSON from text
    json_str = extract_json_from_text(judge_response)
    
    if not json_str:
        logger.debug(f"Could not extract JSON from response: {judge_response[:200]}")
        return None
    
    # Clean JSON string
    json_str = clean_json_string(json_str)
    
    # If truncated, try to repair it
    if is_truncated:
        json_str = repair_truncated_json(json_str)
    
    # Try to parse JSON
    try:
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error after cleaning: {e}\nJSON string: {json_str[:500]}")
        
        # Try repairing truncated JSON if we haven't already
        if not is_truncated:
            try:
                json_str_repaired = repair_truncated_json(json_str)
                result = json.loads(json_str_repaired)
                return result
            except json.JSONDecodeError:
                pass
        
        # Try one more time with more aggressive cleaning
        try:
            # Remove any non-printable characters except newlines
            json_str_clean = ''.join(char for char in json_str if char.isprintable() or char in '\n\r\t')
            result = json.loads(json_str_clean)
            return result
        except json.JSONDecodeError:
            # Last attempt: try repairing again with cleaned string
            try:
                json_str_repaired = repair_truncated_json(json_str_clean)
                result = json.loads(json_str_repaired)
                return result
            except json.JSONDecodeError:
                # Final fallback: try to extract partial fields using regex
                logger.debug("JSON parsing failed completely, attempting partial field extraction")
                partial_result = extract_partial_json_fields(json_str_clean, response_type)
                if partial_result:
                    logger.debug(f"Extracted partial fields: {list(partial_result.keys())}")
                    return partial_result
                return None


def get_strict_json_prompt(original_prompt: str) -> str:
    """Create a stricter prompt that emphasizes JSON-only output
    
    Args:
        original_prompt: Original evaluation prompt
        
    Returns:
        Stricter prompt with JSON-only emphasis
    """
    return f"""{original_prompt}

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON.
Do not use markdown code blocks. Do not add explanations outside the JSON.
The response must be parseable JSON that starts with {{ and ends with }}.
"""


def get_concise_json_prompt(original_prompt: str) -> str:
    """Create a prompt that requests concise JSON output to avoid truncation
    
    Args:
        original_prompt: Original evaluation prompt
        
    Returns:
        Prompt with concise output instructions
    """
    return f"""{original_prompt}

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON.
Do not use markdown code blocks. Do not add explanations outside the JSON.
The response must be parseable JSON that starts with {{ and ends with }}.

IMPORTANT: Keep your response concise to avoid truncation:
- Limit explanations to 2-3 sentences maximum
- For claim lists (SupportedClaims/UnsupportedClaims), include only the 3-5 most important items per category
- Be brief but accurate
"""


def get_relevance_evaluation_prompt(question: str, answer: str) -> str:
    """Generate prompt for relevance evaluation
    
    Args:
        question: User question
        answer: Generated answer
        
    Returns:
        Evaluation prompt
    """
    prompt = f"""
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation as a valid JSON object. Your response must be ONLY valid JSON that starts with {{ and ends with }}.
Do not include any text before or after the JSON. Do not use markdown code blocks.

{{
"Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
"Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()
    return prompt


def get_factual_accuracy_prompt(question: str, answer: str, retrieved_documents: List[str]) -> str:
    """Generate prompt for factual accuracy evaluation
    
    Args:
        question: User question
        answer: Generated answer
        retrieved_documents: List of retrieved document passages
        
    Returns:
        Evaluation prompt
    """
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_documents)])
    
    prompt = f"""
You are an expert evaluator for a RAG system.
Your task is to evaluate the factual accuracy of the generated answer based on the retrieved source documents.
You will classify the answer as "FACTUALLY_CORRECT", "PARTIALLY_CORRECT", or "FACTUALLY_INCORRECT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Retrieved Source Documents:
{context}

Please analyze whether the facts stated in the answer are supported by the retrieved documents.
Consider:
1. Are the claims in the answer supported by the source documents?
2. Are there any factual errors or contradictions?
3. Is the answer missing important information from the sources?

Provide your evaluation as a valid JSON object. Your response must be ONLY valid JSON that starts with {{ and ends with }}.
Do not include any text before or after the JSON. Do not use markdown code blocks.

Keep your response concise:
- Limit explanations to 2-3 sentences maximum
- For claim lists, include only the 3-5 most important items per category

{{
"FactualAccuracy": "FACTUALLY_CORRECT" | "PARTIALLY_CORRECT" | "FACTUALLY_INCORRECT",
"Explanation": "[Provide a brief explanation for your evaluation]",
"SupportedClaims": ["List of claims that are supported by sources"],
"UnsupportedClaims": ["List of claims that are not supported or contradicted"]
}}
""".strip()
    return prompt


def evaluate_relevance(
    question: str,
    answer: str,
    groq_client: Groq,
    judge_model: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
) -> Dict[str, str]:
    """Evaluate answer relevance using LLM-as-a-judge
    
    Args:
        question: User question
        answer: Generated answer
        groq_client: Groq client instance
        judge_model: Model to use for judging
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with relevance score and explanation
    """
    prompt = get_relevance_evaluation_prompt(question, answer)
    
    for attempt in range(max_retries):
        try:
            # Use stricter prompt on retries
            current_prompt = get_strict_json_prompt(prompt) if attempt > 0 else prompt
            
            response = groq_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=0.0,  # Deterministic evaluation
                max_tokens=32768,  # Very high limit - API will cap based on available context window
            )
            
            judge_response = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            is_truncated = (finish_reason == "length")
            
            # Check if response was truncated
            if is_truncated:
                logger.warning(f"Response was truncated (finish_reason=length). Response length: {len(judge_response)} chars. Attempt: {attempt + 1}/{max_retries}")
                # If truncated and we have retries left, try with concise prompt
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying with concise prompt to avoid truncation (attempt {attempt + 1}/{max_retries})")
                    current_prompt = get_concise_json_prompt(prompt)
                    time.sleep(0.5)
                    continue
            
            # Parse JSON response with robust extraction (pass truncation flag)
            result = parse_judge_response(judge_response, "relevance", is_truncated=is_truncated)
            
            if result is not None:
                # Validate required fields
                if "Relevance" in result:
                    return {
                        "relevance": result.get("Relevance", "PARTLY_RELEVANT"),
                        "explanation": result.get("Explanation", ""),
                    }
            
            # If parsing failed and we have retries left, try again
            if attempt < max_retries - 1:
                logger.debug(f"Failed to parse relevance response (attempt {attempt + 1}/{max_retries}), retrying...")
                # Use concise prompt on retry to avoid truncation
                if attempt == 0:
                    current_prompt = get_concise_json_prompt(prompt)
                time.sleep(0.5)  # Small delay between retries
                continue
            else:
                # Last attempt failed - raise exception to be caught by outer handler
                truncation_note = " (truncated)" if finish_reason == "length" else ""
                raise ValueError(f"Failed to parse judge response after {max_retries} attempts{truncation_note}. Response: {judge_response[:200]}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Error evaluating relevance (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(0.5)  # Small delay between retries
                continue
            else:
                logger.error(f"Error evaluating relevance after {max_retries} attempts: {e}")
                raise
    
    # This should never be reached, but just in case
    raise ValueError("Failed to evaluate relevance after all retries")


def evaluate_factual_accuracy(
    question: str,
    answer: str,
    retrieved_documents: List[str],
    groq_client: Groq,
    judge_model: str = "llama-3.3-70b-versatile",
    max_retries: int = 3,
) -> Dict[str, any]:
    """Evaluate factual accuracy using LLM-as-a-judge
    
    Args:
        question: User question
        answer: Generated answer
        retrieved_documents: List of retrieved document passages
        groq_client: Groq client instance
        judge_model: Model to use for judging
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with factual accuracy score and details
    """
    prompt = get_factual_accuracy_prompt(question, answer, retrieved_documents)
    
    # Check if prompt is large and use concise version proactively
    prompt_tokens = count_tokens(prompt, judge_model)
    if prompt_tokens > LARGE_PROMPT_THRESHOLD:
        logger.debug(f"Large prompt detected ({prompt_tokens} tokens), using concise prompt proactively")
        prompt = get_concise_json_prompt(prompt)
    
    for attempt in range(max_retries):
        try:
            # Use stricter prompt on retries
            current_prompt = get_strict_json_prompt(prompt) if attempt > 0 else prompt
            
            response = groq_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=0.0,  # Deterministic evaluation
                max_tokens=65536,  # Very high limit for factual accuracy to handle long lists of claims and detailed explanations - API will cap based on available context window
            )
            
            judge_response = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            is_truncated = (finish_reason == "length")
            
            # Check if response was truncated
            if is_truncated:
                logger.warning(f"Response was truncated (finish_reason=length). Response length: {len(judge_response)} chars. Attempt: {attempt + 1}/{max_retries}")
                # If truncated and we have retries left, try with concise prompt
                if attempt < max_retries - 1:
                    logger.debug(f"Retrying with concise prompt to avoid truncation (attempt {attempt + 1}/{max_retries})")
                    current_prompt = get_concise_json_prompt(prompt)
                    time.sleep(0.5)
                    continue
            
            # Parse JSON response with robust extraction (pass truncation flag)
            result = parse_judge_response(judge_response, "factual_accuracy", is_truncated=is_truncated)
            
            if result is not None:
                # Validate required fields
                if "FactualAccuracy" in result:
                    return {
                        "factual_accuracy": result.get("FactualAccuracy", "PARTIALLY_CORRECT"),
                        "explanation": result.get("Explanation", ""),
                        "supported_claims": result.get("SupportedClaims", []),
                        "unsupported_claims": result.get("UnsupportedClaims", []),
                    }
            
            # If parsing failed and we have retries left, try again
            if attempt < max_retries - 1:
                logger.debug(f"Failed to parse factual accuracy response (attempt {attempt + 1}/{max_retries}), retrying...")
                # Use concise prompt on retry to avoid truncation
                if attempt == 0:
                    current_prompt = get_concise_json_prompt(prompt)
                time.sleep(0.5)  # Small delay between retries
                continue
            else:
                # Last attempt failed - raise exception to be caught by outer handler
                truncation_note = " (truncated)" if finish_reason == "length" else ""
                raise ValueError(f"Failed to parse judge response after {max_retries} attempts{truncation_note}. Response: {judge_response[:200]}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Error evaluating factual accuracy (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(0.5)  # Small delay between retries
                continue
            else:
                logger.error(f"Error evaluating factual accuracy after {max_retries} attempts: {e}")
                raise
    
    # This should never be reached, but just in case
    raise ValueError("Failed to evaluate factual accuracy after all retries")


def score_to_numeric(score: str, metric_type: str = "relevance") -> float:
    """Convert categorical score to numeric value
    
    Args:
        score: Categorical score string
        metric_type: Type of metric ("relevance" or "factual_accuracy")
        
    Returns:
        Numeric score (0.0 to 1.0)
    """
    if metric_type == "relevance":
        mapping = {
            "NON_RELEVANT": 0.0,
            "PARTLY_RELEVANT": 0.5,
            "RELEVANT": 1.0,
        }
    elif metric_type == "factual_accuracy":
        mapping = {
            "FACTUALLY_INCORRECT": 0.0,
            "PARTIALLY_CORRECT": 0.5,
            "FACTUALLY_CORRECT": 1.0,
        }
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    
    return mapping.get(score.upper(), 0.5)

