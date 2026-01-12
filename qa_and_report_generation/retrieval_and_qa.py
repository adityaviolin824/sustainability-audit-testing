import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from tqdm import tqdm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_exponential

# Custom Imports
from utils.logger import logging
from utils.exception import CustomException
from retrieval_and_postprocessing.retrieval_full_pipeline import AdvancedRAGRetrievalEngine

logger = logging.getLogger(__name__)

# --- CORE RAG UTILS ---

def make_rag_messages(question: str, chunks: List[Any], history: Optional[List[Dict]] = None) -> List[Dict]:
    """Formats the system prompt, context, and history for the LLM."""
    try:
        history = history or []
        context_str = ""
        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get('page', 'N/A')
            context_str += f"\n--- Context Chunk {i+1} (Page {page}) ---\n{chunk.page_content}\n"

        system_content = (
            "You are a precise AI assistant. Answer the user's question using ONLY the provided context. "
            "If the information is not present, respond exactly: 'Not disclosed in the report.' "
            "Always cite specific Page numbers for every extraction."
        )
        
        return [
            {"role": "system", "content": system_content},
            *history,
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {question}"}
        ]
    except Exception as e:
        raise CustomException(e, sys)


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def answer_batch(
    batch_name: str,
    questions: List[Dict],
    chunks: List[Any],
    model: str = "openai/gpt-4.1-mini"
) -> Dict[str, Any]:
    """Extracts information for all questions in a batch using a single LLM call."""
    try:
        combined_prompt = (
            f"You are performing an ESG audit extraction task for the section: {batch_name}.\n"
            "STRICT: Use ONLY the provided context. Cite exact page numbers. No summaries.\n\n"
            "ANSWER FORMAT PER QUESTION:\n"
            "<QUESTION_ID>:\n"
            "Answer: <verbatim statement or 'Not disclosed in the report.'>\n"
            "Page: <page number or 'N/A'>\n"
            "Evidence: \"<exact quote>\"\n\n"
            "QUESTIONS:\n"
        )

        for q in questions:
            combined_prompt += f"{q['id']}. {q['question']}\n"

        messages = make_rag_messages(question=combined_prompt, chunks=chunks)
        response = completion(model=model, messages=messages)
        
        return {
            "batch": batch_name,
            "raw_answer": response.choices[0].message.content.strip(),
            "num_chunks_used": len(chunks)
        }
    except Exception as e:
        raise CustomException(e, sys)


# --- DATA HANDLING ---

def load_questions_by_batch(path: str) -> Dict[str, List[Dict]]:
    """Loads questions from JSONL and groups them by the 'batch' key."""
    try:
        batches: Dict[str, List[Dict]] = {}
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Question file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                q = json.loads(line)
                batches.setdefault(q["batch"], []).append(q)
        return batches
    except Exception as e:
        raise CustomException(e, sys)


# --- EXECUTION PIPELINE ---

def run_esg_batch_extraction(
    config_path: str,
    db_path: str,
    question_path: str,
    run_dir: Path,
    model: str = "openai/gpt-4.1-mini",
    debug_filename: str = "batchwise_responses_audit.md",
    answers_filename: str = "batchwise_answers_only.txt"
):
    """
    Main pipeline execution for batched RAG extraction.
    Outputs a formatted Markdown audit log including retrieved context chunks.
    """
    try:
        logging.info("Initializing Advanced RAG Retrieval Engine...")
        engine = AdvancedRAGRetrievalEngine(config_path=Path(config_path), db_path=Path(db_path))
        
        debug_md_path = run_dir / debug_filename
        answers_txt_path = run_dir / answers_filename
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        debug_md_path.write_text("# ESG Audit Extraction - Process Log\n", encoding="utf-8")
        answers_txt_path.write_text("", encoding="utf-8")

        batches = load_questions_by_batch(question_path)
        logging.info(f"Loaded {len(batches)} batches for processing.")

        for batch_name, questions in batches.items():
            logging.info(f"Starting Processing: Batch [{batch_name}] with {len(questions)} questions.")
            
            all_chunks = []
            seen_chunk_keys = set()

            for q in tqdm(questions, desc=f"Retrieving [{batch_name}]", unit="q"):
                context, _ = engine.get_context_advanced(q["question"])
                for chunk in context:
                    dedup_key = (chunk.metadata.get("page", "N/A"), hash(chunk.page_content))
                    if dedup_key not in seen_chunk_keys:
                        seen_chunk_keys.add(dedup_key)
                        all_chunks.append(chunk)

            logging.info(f"Retrieved {len(all_chunks)} unique context chunks for batch {batch_name}.")

            try:
                result = answer_batch(batch_name, questions, all_chunks, model=model)
                raw_answer = result["raw_answer"]

                # 1. Write formatted Markdown for human audit
                with open(debug_md_path, "a", encoding="utf-8") as f:
                    f.write(f"\n## Batch: **{batch_name}**\n")
                    f.write(f"> **LLM Extraction Result:**\n>\n")
                    f.write(f"{raw_answer}\n")
                    
                    # NEW: Appending all extracted chunks for this batch
                    f.write(f"\n### 📚 Retrieved Context Chunks (Total: {len(all_chunks)})\n")
                    for i, chunk in enumerate(all_chunks):
                        page = chunk.metadata.get('page', 'N/A')
                        f.write(f"\n#### Chunk {i+1} (Page {page})\n")
                        f.write(f"```text\n{chunk.page_content}\n```\n")
                    
                    f.write("\n---\n")
                
                # 2. Write raw text for downstream machine parsing
                with open(answers_txt_path, "a", encoding="utf-8") as f:
                    f.write(f"\n{raw_answer}\n\n") 

                logging.info(f"Successfully processed and saved batch: {batch_name}")
                
            except Exception as e:
                err = CustomException(e, sys)
                logging.error(f"Generation failed for batch {batch_name}: {err}")

    except Exception as e:
        raise CustomException(e, sys)
