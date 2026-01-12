import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm


import litellm
litellm.set_verbose = False
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

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

