import sys
import openai
from pathlib import Path
from typing import List, Dict, Any

# Third-party isolated imports
from chromadb import PersistentClient

# Your project utility imports
from utils.logger import logging
from utils.exception import CustomException

# Initialize Logger
logger = logging.getLogger(__name__)

def create_embeddings_direct(chunks: List[Dict[str, Any]], db_path: Path, collection_name: str, embedding_model: str):
    """
    Directly embeds chunks using OpenAI and stores them in a persistent ChromaDB.
    
    Args:
        chunks: List of dictionaries containing "text" and "metadata".
        db_path: Pathlib object pointing to the unique run directory for the database[cite: 9].
        collection_name: Name of the collection to create/update.
        embedding_model: The OpenAI model string (e.g., 'text-embedding-3-small').
    """
    try:
        # 1. Initialize Persistent Chroma Client
        # Using a persistent client ensures the reporting boundary is maintained[cite: 10].
        chroma = PersistentClient(path=str(db_path))
        
        # 2. Collection Management
        # Delete existing collection if it exists to ensure a fresh build for the specific run
        existing_collections = [c.name for c in chroma.list_collections()]
        if collection_name in existing_collections:
            chroma.delete_collection(collection_name)
            logger.info(f"Cleared existing collection: {collection_name}")
            
        collection = chroma.create_collection(name=collection_name)

        # 3. Data Preparation
        # SEBI requires clear and complete responses; metadata preservation is key[cite: 14].
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        # Generate IDs based on chunk index or provided chunk_id
        ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

        # 4. Direct Batch Embedding Call
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        response = openai.embeddings.create(
            model=embedding_model,
            input=texts
        )
        
        # Extract numerical vectors from OpenAI response
        vectors = [e.embedding for e in response.data]

        # 5. Persist to Vectorstore
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Successfully finalized vectorstore with {collection.count()} records at {db_path}.")
        return collection

    except Exception as e:
        # Wrap in your custom exception handler for the pipeline
        raise CustomException(e, sys)