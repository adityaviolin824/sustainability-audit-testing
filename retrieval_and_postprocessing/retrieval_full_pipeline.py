# retrieval_and_postprocessing\retrieval_full_pipeline.py

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Component Imports
from retrieval_and_postprocessing.retrieval_functions import RAGRetrievalEngine, Result
from retrieval_and_postprocessing.llm_reranking_and_query_processing import (
    rewrite_query, 
    rerank, 
    merge_chunks
)

# Utility Imports
from utils.logger import logging
from utils.exception import CustomException
from utils.read_yaml import read_yaml

logger = logging.getLogger(__name__)

class AdvancedRAGRetrievalEngine(RAGRetrievalEngine):
    """
    Extends base engine using centralized YAML configuration.
    """
    
    def __init__(self, config_path: Path, db_path: Path):
        # 1. Load Config
        self.config = read_yaml(config_path)
        
        # 2. Initialize Parent (Base Engine)
        super().__init__(
            db_path=db_path,
            collection_name=self.config.vectorstore.collection_name,
            embedding_model=self.config.vectorstore.embedding_model
        )
        logger.info("Advanced Engine initialized with YAML configuration.")

    def get_context_advanced(
        self, 
        question: str, 
        history: List[Dict] = []
    ) -> Tuple[List[Result], str]:
        """
        Retrieves context based on parameters defined in the master config.
        """
        try:
            cfg = self.config  # For cleaner access
            
            # 1. Query Processing
            target_queries = [question]
            expanded_query_display = "N/A (Original only)"
            
            if cfg.pipeline_logic.process_query:
                rewritten = rewrite_query(
                    question, 
                    model=cfg.models.query_expansion_model, 
                    history=history
                )
                target_queries.append(rewritten)
                expanded_query_display = rewritten
                logger.info(f"Query Processed: {rewritten}")

            # 2. Dual Retrieval & Merging
            all_candidate_chunks = []
            for q in target_queries:
                chunks = self.fetch_context_unranked(q, n_results=cfg.retrieval.initial_k)
                if not all_candidate_chunks:
                    all_candidate_chunks = chunks
                else:
                    all_candidate_chunks = merge_chunks(all_candidate_chunks, chunks)
            
            # 3. Reranking
            if cfg.pipeline_logic.use_reranking:
                logger.info(f"Reranking {len(all_candidate_chunks)} candidates...")
                final_chunks = rerank(
                    question, 
                    all_candidate_chunks, 
                    model=cfg.models.reranking_model
                )
                return final_chunks[:cfg.retrieval.final_k], expanded_query_display
            
            # 4. Fallback
            return all_candidate_chunks[:cfg.retrieval.final_k], expanded_query_display

        except Exception as e:
            raise CustomException(e, sys)



# --- EXECUTION DEMO ---

if __name__ == "__main__":
    # --- PATHS ---
    CONFIG_PATH = Path("config/retrieval_master_config.yaml")
    # This remains dynamic as it depends on your current ingestion run
    RUN_DIR = Path(r"runs\Infosys BRSR 2024_20260107_085246")
    DB_PATH = RUN_DIR / "chroma_db"
    
    # 1. Initialize Engine using YAML
    engine = AdvancedRAGRetrievalEngine(config_path=CONFIG_PATH, db_path=DB_PATH)

    # 2. Run Retrieval
    test_q = "Which national or international certifications has the company adopted?"
    context, used_query = engine.get_context_advanced(test_q)

    # 3. Export to File (using paths from config)
    output_file = Path(engine.config.output.demo_export_path)
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w", encoding=engine.config.output.encoding) as f:
        f.write(f"RAG AUDIT REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Query: {test_q}\n")
        f.write(f"Expanded Query: {used_query}\n")
        f.write("=" * 60 + "\n\n")

        for i, chunk in enumerate(context):
            source = engine._get_safe_source(chunk.metadata)
            f.write(f"RANK #{i+1} [Page: {chunk.metadata.get('page')} | Principle: {chunk.metadata.get('principle')}]\n")
            f.write(f"Source: {source}\n")
            f.write(f"Content:\n{chunk.page_content}\n")
            f.write("-" * 40 + "\n\n")

    print(f"Audit Results exported successfully to: {output_file}")