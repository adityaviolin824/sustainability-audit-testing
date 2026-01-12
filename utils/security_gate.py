import torch
import gc
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import ahocorasick

# Force PyTorch to use minimal resources
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class SecurityGate:
    def __init__(self, blacklist: List[str], threshold: float = 0.85):
        # 1. Deterministic Layer (O(n))
        self.automaton = ahocorasick.Automaton()
        for idx, word in enumerate(blacklist):
            self.automaton.add_word(word.lower(), (idx, word))
        self.automaton.make_automaton()
        
        # 2. Semantic Layer (Extreme Small Model - ~30MB)
        # Using 'paraphrase-albert-small-v2' or 'paraphrase-MiniLM-L3-v2' 
        self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.model.eval() # Set to evaluation mode
        
        with torch.no_grad():
            self.blacklist_embeddings = self.model.encode(blacklist, convert_to_tensor=True)
        
        self.threshold = threshold

    @torch.inference_mode()
    def is_malicious(self, text: str) -> Tuple[bool, str]:
        text_lower = text.lower()

        # LAYER 1: Aho-Corasick
        for _, (_, original_value) in self.automaton.iter(text_lower):
            return True, f"Deterministic Match: {original_value}"

        # LAYER 2: Semantic Similarity
        query_embedding = self.model.encode(text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_embedding, self.blacklist_embeddings)
        max_score = float(cosine_scores.max())

        # Cleanup RAM immediately
        del query_embedding, cosine_scores
        gc.collect()

        if max_score > self.threshold:
            return True, f"Semantic Match (Score: {max_score:.2f})"

        return False, ""