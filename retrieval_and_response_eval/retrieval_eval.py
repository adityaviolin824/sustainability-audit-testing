import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm

# --- 0. DISABLE LITELLM LOGGING ---
# This must happen before other logic to prevent initial info logs
import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True
# Silence the underlying logger completely
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Component Imports
from retrieval_and_postprocessing.retrieval_full_pipeline import AdvancedRAGRetrievalEngine
from retrieval_and_response_eval.test import TestQuestion, load_tests

# -----------------------------
# Data Schema
# -----------------------------
class RetrievalEval(BaseModel):
    """Metrics that evaluate how well relevant context was retrieved."""
    mrr: float = Field(description="Mean Reciprocal Rank across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="How many keywords appeared in top-k results")
    total_keywords: int = Field(description="Total number of keywords")
    keyword_coverage: float = Field(description="Percentage of keywords successfully retrieved")

# -----------------------------
# Core Evaluation Logic
# -----------------------------
class RetrievalEvaluator:
    """ Handles the mathematical scoring of retrieval quality. """
    def __init__(self, engine: AdvancedRAGRetrievalEngine):
        self.engine = engine

    def _calculate_mrr(self, keyword: str, retrieved_docs: list) -> float:
        """Return reciprocal rank of the first document containing the keyword."""
        keyword_lower = keyword.lower()
        for rank, doc in enumerate(retrieved_docs, start=1):
            if keyword_lower in doc.page_content.lower():
                return 1.0 / rank
        return 0.0

    def _calculate_dcg(self, relevances: list[int], k: int) -> float:
        """Compute Discounted Cumulative Gain up to rank k."""
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            dcg += relevances[i] / math.log2(i + 2)
        return dcg

    def _calculate_ndcg(self, keyword: str, retrieved_docs: list, k: int = 10) -> float:
        """Compute nDCG for a keyword using binary relevance."""
        keyword_lower = keyword.lower()
        relevances = [
            1 if keyword_lower in doc.page_content.lower() else 0
            for doc in retrieved_docs[:k]
        ]
        dcg = self._calculate_dcg(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self._calculate_dcg(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0

    def run_evaluation(self, test: TestQuestion, k: int = 10) -> RetrievalEval:
        """Runs the retrieval metrics for a single test case."""
        # Calls the full pipeline (Expansion -> Search -> Merge -> Rerank)
        retrieved_docs, _ = self.engine.get_context_advanced(test.question)

        mrr_scores = [self._calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
        ndcg_scores = [self._calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]

        keywords_found = sum(1 for score in mrr_scores if score > 0)
        total_keywords = len(test.keywords)
        
        return RetrievalEval(
            mrr=sum(mrr_scores) / total_keywords if total_keywords > 0 else 0.0,
            ndcg=sum(ndcg_scores) / total_keywords if total_keywords > 0 else 0.0,
            keywords_found=keywords_found,
            total_keywords=total_keywords,
            keyword_coverage=(keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0,
        )

# -----------------------------
# Modular Execution Block
# -----------------------------
if __name__ == "__main__":
    # --- 1. ENVIRONMENT & PATHS ---
    load_dotenv(override=True)
    
    CONFIG_PATH = Path("config/retrieval_master_config.yaml")
    RUN_DIR = Path(r"runs\Reliance BRSR 2022_20260101_145411")
    DB_PATH = RUN_DIR / "chroma_db"
    SUMMARY_EXPORT = Path("testing/batch_retrieval_results.json")

    # --- 2. INITIALIZATION ---
    engine = AdvancedRAGRetrievalEngine(config_path=CONFIG_PATH, db_path=DB_PATH)
    evaluator = RetrievalEvaluator(engine=engine)
    tests = load_tests()

    # --- 3. BATCH RUN WITH TQDM ---
    print(f"🚀 Starting Retrieval Evaluation for {len(tests)} cases...")
    print("=" * 60)

    all_results = []
    
    # tqdm creates a progress bar and allows us to update description text
    pbar = tqdm(tests, desc="Evaluating RAG Retrieval", unit="test")
    
    for i, test in enumerate(pbar):
        try:
            metrics = evaluator.run_evaluation(test)
            all_results.append({
                "id": i,
                "category": test.category,
                "question": test.question,
                "metrics": metrics.model_dump()
            })
            
            # Update the progress bar with real-time stats
            pbar.set_postfix({
                "category": test.category[:10],
                "MRR": f"{metrics.mrr:.3f}",
                "Cov": f"{metrics.keyword_coverage:.0f}%"
            })
            
        except Exception as e:
            # We don't want to break the progress bar on error
            tqdm.write(f"❌ Failed Test #{i}: {e}")



    # --- 4. EXPORT SUMMARY (TXT TABLE) ---
    SUMMARY_EXPORT = Path("testing/batch_retrieval_results.txt")
    SUMMARY_EXPORT.parent.mkdir(exist_ok=True)

    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    num_tests = len(all_results)

    with open(SUMMARY_EXPORT, "w", encoding="utf-8") as f:
        # Header
        f.write(
            "ID | Category     | MRR   | nDCG  | KW Found | KW Total | Coverage % | Question\n"
        )
        f.write("-" * 90 + "\n")

        for r in all_results:
            m = r["metrics"]
            total_mrr += m["mrr"]
            total_ndcg += m["ndcg"]
            total_coverage += m["keyword_coverage"]

            f.write(
                f"{r['id']:>2} | "
                f"{r['category']:<12} | "
                f"{m['mrr']:<5.3f} | "
                f"{m['ndcg']:<5.3f} | "
                f"{m['keywords_found']:^8} | "
                f"{m['total_keywords']:^8} | "
                f"{m['keyword_coverage']:^10.1f} | "
                f"{r['question']}\n"
            )

        # Overall metrics
        f.write("\n" + "=" * 90 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 90 + "\n")
        f.write(f"Total Tests          : {num_tests}\n")
        f.write(f"Average MRR          : {total_mrr / num_tests:.3f}\n")
        f.write(f"Average nDCG         : {total_ndcg / num_tests:.3f}\n")
        f.write(f"Average Coverage (%) : {total_coverage / num_tests:.1f}\n")
        f.write("=" * 90 + "\n")

    print(f"\n✅ Evaluation complete. TXT report saved to: {SUMMARY_EXPORT}")
