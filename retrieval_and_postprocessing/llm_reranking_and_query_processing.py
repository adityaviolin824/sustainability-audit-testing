import sys
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from litellm import completion
from tenacity import retry, wait_exponential

# Your project utility imports
from utils.logger import logging
from utils.exception import CustomException

logger = logging.getLogger(__name__)

# --- DATA MODELS ---

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# --- LOGIC FUNCTIONS ---

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def rewrite_query(question: str, model: str, history: List[Dict] = []) -> str:
    """
    Expert ESG Auditor rewriting logic.
    Expands queries with technical synonyms and regulatory frameworks (BRSR/NGRBC).
    """
    try:
        system_msg = (
            "You are an expert ESG Auditor. Rewrite user questions into "
            "precise search queries for a BRSR (Business Responsibility and Sustainability Report). "
            "Expand technical terms (e.g., CSR, GHG, Scope 3) using BRSR and NGRBC-aligned language. "
            "Keep the query concise. Respond ONLY with the refined query."
        )

        # NOTE:
        # Conversation history is intentionally NOT used for query expansion.
        # Retrieval should remain intent-focused and evidence-driven.
        # history is kept in the function signature to preserve pipeline compatibility.

        user_msg = question  # history intentionally ignored

        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Query rewriting failed: {e}")
        return question  # Fallback to original


@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def rerank(question: str, chunks: List[Result], model: str) -> List[Result]:
    """
    Senior Sustainability Auditor re-ranking logic.
    Prioritizes quantitative data, Principle-specific tables, and evidence over narrative.
    """
    if not chunks:
        return []

    try:
        system_prompt = (
            "You are a Senior Sustainability Auditor. Rank document chunks "
            "based on their ability to provide a FACTUAL and QUANTITATIVE answer. "
            "\nPriority Criteria:\n"
            "1. Chunks with specific metrics, tables, or financial figures.\n"
            "2. Chunks explicitly referencing SEBI BRSR Principles.\n"
            "3. Chunks with specific policy names or web links.\n"
            "Ignore boilerplate legal disclaimers. Respond ONLY with the ranked chunk IDs."
        )
        
        user_prompt = f"Target Question: {question}\n\n"
        for idx, chunk in enumerate(chunks):
            # Include principle metadata to help the LLM contextualize the snippet
            princ = chunk.metadata.get('principle', 'Unknown Section')
            user_prompt += f"# CHUNK ID {idx + 1} (Section: {princ}):\n{chunk.page_content}\n\n"
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = completion(
            model=model, 
            messages=messages, 
            response_format=RankOrder
        )
        
        order = RankOrder.model_validate_json(response.choices[0].message.content).order
        
        # Safe re-sorting (guarding against hallucinated IDs)
        return [chunks[i - 1] for i in order if 0 < i <= len(chunks)]
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        return chunks

def merge_chunks(chunks_a: List[Result], chunks_b: List[Result]) -> List[Result]:
    """
    Standardizes and de-duplicates results from dual-retrieval.
    """
    merged = chunks_a[:]
    
    existing_keys = {
        (
            c.page_content.strip(),
            c.metadata.get("page"),
            c.metadata.get("source_id"),
        )
        for c in chunks_a
    }
    
    for chunk in chunks_b:
        key = (
            chunk.page_content.strip(),
            chunk.metadata.get("page"),
            chunk.metadata.get("source_id"),
        )
        if key not in existing_keys:
            merged.append(chunk)
            existing_keys.add(key)
            
    return merged
