import json
import sys
from pathlib import Path
from typing import Dict, List, Any

from pydantic import BaseModel, Field
from litellm import completion

# Custom Utility Imports 
from utils.logger import logging
from utils.exception import CustomException

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Evaluation Schema
# --------------------------------------------------
class AnswerEval(BaseModel):
    """LLM-as-a-judge scoring schema."""
    feedback: str = Field(description="Qualitative comparison vs reference")
    accuracy: float = Field(description="Factual correctness (1-5)")
    completeness: float = Field(description="Information coverage (1-5)")
    relevance: float = Field(description="Directness/Conciseness (1-5)")

# --------------------------------------------------
# Evaluation Logic
# --------------------------------------------------
def evaluate_record(gen_rec: Dict, ref_rec: Dict, model: str) -> AnswerEval:
    """Evaluates a single extracted record against the ground truth."""
    try:
        judge_messages = [
            {
                "role": "system",
                "content": "You are a strict ESG auditor. Only give 5/5 for perfect factual matches."
            },
            {
                "role": "user",
                "content": f"""Question: {ref_rec['question']}
Extracted Answer: {gen_rec.get('summary', 'Not disclosed')}
Key Facts Found: {gen_rec.get('key_facts', [])}
Reference (Ground Truth): {ref_rec['reference_answer']}

Evaluate on accuracy, completeness, and relevance (1-5).
If the extraction missed a value present in the reference, accuracy must be low."""
            }
        ]

        response = completion(
            model=model,
            messages=judge_messages,
            response_format=AnswerEval
        )
        
        return AnswerEval.model_validate_json(response.choices[0].message.content)
    except Exception as e:
        raise CustomException(e, sys)

# --------------------------------------------------
# Execution Script
# --------------------------------------------------
def main(gen_path: str, ref_path: str, output_path: str):
    try:
        # 1. Load Data
        gen_data = json.loads(Path(gen_path).read_text(encoding="utf-8"))
        
        ref_map = {}
        with open(ref_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                ref_map[obj["id"]] = obj

        results = []
        logging.info(f"Evaluating {len(gen_data)} records...")

        # 2. Process
        for rec in gen_data:
            m_id = rec["metric_id"]
            if m_id in ref_map:
                score = evaluate_record(rec, ref_map[m_id], "gpt-5-nano")
                results.append({
                    "metric_id": m_id,
                    "scores": score.model_dump()
                })
                print(f"[{m_id}] Accuracy: {score.accuracy}/5 | Completeness: {score.completeness}/5")

        # 3. Save
        Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logging.info(f"Results saved to {output_path}")

    except Exception as e:
        err = CustomException(e, sys)
        logging.error(f"Evaluation failed: {err}")