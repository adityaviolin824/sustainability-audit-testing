import sys
import openai
from pathlib import Path
from typing import List, Dict, Any

# Third-party isolated imports
from chromadb import PersistentClient

# Your project utility imports
from utils.logger import logging
from utils.exception import CustomException
from utils.read_yaml import read_yaml

# Component Imports
from vectorstore_ingestion.report_data_extraction import ReportIntakePipeline
from vectorstore_ingestion.chunk_preprocessing import chunk_document_final

# Initialize Logger
logger = logging.getLogger(__name__)

def create_embeddings_direct(chunks: List[Dict[str, Any]], db_path: Path, collection_name: str, embedding_model: str):
    """
    Directly embeds chunks using OpenAI in batches and stores them in ChromaDB.
    """
    try:
        chroma = PersistentClient(path=str(db_path))
        
        existing_collections = [c.name for c in chroma.list_collections()]
        if collection_name in existing_collections:
            chroma.delete_collection(collection_name)
            logger.info(f"Cleared existing collection: {collection_name}")
            
        collection = chroma.create_collection(name=collection_name)

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]

        # --- BATCHING LOGIC START ---
        # 100 chunks is a safe default to avoid the 300k token limit
        batch_size = 100 
        vectors = []
        
        logger.info(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            response = openai.embeddings.create(
                model=embedding_model,
                input=batch_texts
            )
            
            # Extract and append vectors from this batch
            batch_vectors = [e.embedding for e in response.data]
            vectors.extend(batch_vectors)
            
            logger.info(f"Processed batch {int(i/batch_size) + 1}/{(len(texts)-1)//batch_size + 1}")
        # --- BATCHING LOGIC END ---

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Successfully finalized vectorstore with {collection.count()} records at {db_path}.")
        return collection

    except Exception as e:
        raise CustomException(e, sys)

def run_ingestion_pipeline(input_pdf_path: Path, config_path: Path, run_paths: Dict[str, Path]):
    """
    Data processing pipeline. Orchestration (folder creation) is handled by the caller (API).
    """
    try:
        config = read_yaml(config_path)
        
        # Silence libraries as per config
        if hasattr(config, 'logging'):
            for lib in config.logging.silence_libraries:
                logging.getLogger(lib).setLevel(logging.WARNING)
        
        logger.info(f"--- STARTING PROCESSING: {input_pdf_path.name} ---")

        # 1. STAGE 1: Report Ingestion (PDF -> Text)
        params = config.processing_params
        pipeline = ReportIntakePipeline(hard_page_limit=params.hard_page_limit)
        pipeline.run_report_ingestion(
            file_path=str(input_pdf_path), 
            output_txt=str(run_paths["formatted_txt"]),
            start_page=1, 
            batch_size=params.batch_size,
            max_workers=params.max_workers
        )
        logger.info("Stage 1 complete: Text extracted.")

        # 2. STAGE 2: Contextualized Chunking
        if not run_paths["formatted_txt"].exists():
            raise FileNotFoundError(f"Formatted text missing: {run_paths['formatted_txt']}")

        full_text = run_paths["formatted_txt"].read_text(encoding="utf-8")
        chunks = chunk_document_final(full_text, input_pdf_path.name)
        
        # Debug output for verification
        with run_paths["chunks_debug"].open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write(f"CHUNK ID: {ch.get('chunk_id')}\n{ch['text']}\n\n")
        logger.info(f"Stage 2 complete: {len(chunks)} chunks created.")

        # 3. STAGE 3: Vectorstore Creation
        vs_cfg = config.vectorstore_params
        create_embeddings_direct(
            chunks=chunks,
            db_path=run_paths["db_path"],
            collection_name=vs_cfg.collection_name,
            embedding_model=vs_cfg.embedding_model
        )

        logger.info(f"--- PROCESSING SUCCESSFUL ---")
        return True

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    # In a real scenario, the API would generate these
    from datetime import datetime
    
    BASE_DIR = Path(r"runs")
    PDF_FILE = Path(r"reference\sample_documents\Infosys BRSR 2024.pdf")
    CONFIG = Path(r"config\ingestion_master_config.yaml")
    
    # Simulating API orchestration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BASE_DIR / f"{PDF_FILE.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    RUN_PATHS = {
        "formatted_txt": run_dir / "formatted_report.txt",
        "chunks_debug": run_dir / "chunks_preview.txt",
        "db_path": run_dir / "chroma_db"
    }
    
    run_ingestion_pipeline(PDF_FILE, CONFIG, RUN_PATHS)