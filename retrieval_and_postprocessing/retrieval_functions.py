# retrieval_and_postprocessing\retrieval.py

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import openai
from openai import OpenAI
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential
from dotenv import load_dotenv

# Your project utility imports
from utils.logger import logging
from utils.exception import CustomException

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# --- DATA MODELS ---

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

# --- CORE ENGINE ---

class RAGRetrievalEngine:
    def __init__(self, db_path: Path, collection_name: str, embedding_model: str = "text-embedding-3-small"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Initialize Clients
        self.openai_client = OpenAI()
        self.chroma_client = PersistentClient(path=str(db_path))
        
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"Connected to collection '{collection_name}' with {self.collection.count()} records.")
        except Exception as e:
            logger.error(f"Failed to connect to collection {collection_name}: {e}")
            raise CustomException(e, sys)

        # Retry logic for API calls
        self.wait_strategy = wait_exponential(multiplier=1, min=4, max=10)

    def _get_safe_source(self, metadata: Dict) -> str:
        """Helper to handle inconsistent naming in metadata (source vs source_file)."""
        return metadata.get('source') or metadata.get('source_file') or "Unknown Source"

    def fetch_context_unranked(self, question: str, n_results: int = 20) -> List[Result]:
        """Performs raw vector search against the ChromaDB."""
        try:
            # 1. Embed the query
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, 
                input=[question]
            )
            query_vector = response.data[0].embedding

            # 2. Query Vectorstore
            results = self.collection.query(
                query_embeddings=[query_vector], 
                n_results=n_results
            )

            # 3. Format as Pydantic models
            formatted_results = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                formatted_results.append(Result(page_content=doc, metadata=meta))
            
            return formatted_results
        except Exception as e:
            raise CustomException(e, sys)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    def rewrite_query(self, question: str, history: List[Dict] = []) -> str:
        """Rewrites user query to be more specific for Knowledge Base search."""
        message = f"""
        Refine the following user question into a short, precise search query 
        optimized for finding financial and ESG data in a BRSR report.
        Question: {question}
        History: {history}
        Respond ONLY with the refined query text.
        """
        response = completion(
            model="gpt-4.1-nano", 
            messages=[{"role": "user", "content": message}]
        )
        return response.choices[0].message.content

    def get_context(self, question: str, final_k: int = 10) -> List[Result]:
        """Orchestrates query rewriting and retrieval."""
        # Optional: Add Re-ranking logic here later
        refined_q = self.rewrite_query(question)
        logger.info(f"Rewritten Query: {refined_q}")
        
        chunks = self.fetch_context_unranked(refined_q, n_results=final_k)
        return chunks