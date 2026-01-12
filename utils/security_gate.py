import sys
import ahocorasick
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from utils.logger import logging
from utils.exception import CustomException

logger = logging.getLogger(__name__)

class SecurityGate:
    def __init__(self, blacklist: List[str], threshold: float = 0.85):
        # 1. Deterministic Layer (Aho-Corasick)
        self.automaton = ahocorasick.Automaton()
        for idx, word in enumerate(blacklist):
            self.automaton.add_word(word.lower(), (idx, word))
        self.automaton.make_automaton()
        
        # 2. Semantic Layer (Local HuggingFace Model)
        # 2026 standard for high-speed local CPU inference
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.blacklist_embeddings = self.model.encode(blacklist, convert_to_tensor=True)
        self.threshold = threshold
        
        logger.info("Security Gate initialized: Layer 1 (Deterministic) & Layer 2 (Semantic).")

    def is_malicious(self, text: str) -> Tuple[bool, str]:
        """Runs a two-layer check on the provided text."""
        text_lower = text.lower()

        # LAYER 1: Aho-Corasick (O(n) Deterministic)
        for end_index, (insert_order, original_value) in self.automaton.iter(text_lower):
            return True, f"Deterministic Match: {original_value}"

        # LAYER 2: Semantic Similarity (Local CPU)
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.blacklist_embeddings)
        max_score = cosine_scores.max().item()

        if max_score > self.threshold:
            return True, f"Semantic Match (Score: {max_score:.2f})"

        return False, ""