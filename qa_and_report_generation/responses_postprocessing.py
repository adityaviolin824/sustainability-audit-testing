import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from litellm import completion
from utils.logger import logging
from utils.exception import CustomException

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Parser Logic
# --------------------------------------------------

def parse_model_answers(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parses the raw LLM output text into a list of structured dictionaries.
    
    Args:
        raw_text: The raw string content from the batchwise extraction.
        
    Returns:
        A list of dictionaries containing metric_id, answer, page, and evidence.
    """
# --------------------------------------------------
# (Updated Regex)
# --------------------------------------------------

    try:
        records = []
        # FIX: Updated regex to handle start of string (^) or newline (\n)
        blocks = re.split(r"(?:\n|^)(?=[A-Z]{1,3}_[0-9]{2}:)", raw_text.strip())

        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines or not re.match(r"[A-Z]{1,3}_[0-9]{2}:", lines[0]):
                continue

            record = {
                "metric_id": lines[0].replace(":", ""),
                "answer": None,
                "page": None,
                "evidence": None,
            }

            for line in lines[1:]:
                if line.startswith("Answer:"):
                    record["answer"] = line.split(":", 1)[1].strip()
                elif line.startswith("Page:"):
                    record["page"] = line.split(":", 1)[1].strip()
                elif line.startswith("Evidence:"):
                    record["evidence"] = line.split(":", 1)[1].strip()

            records.append(record)
        
        logger.info(f"Parsed {len(records)} records from raw text.")
        return records
    except Exception as e:
        raise CustomException(e, sys)

# --------------------------------------------------
# Consolidation Logic
# --------------------------------------------------

def build_consolidation_prompt(record: Dict[str, Any]) -> str:
    """Constructs the prompt for the second-pass consolidation."""
    return f"""
You are performing a SECOND-PASS ESG CONSOLIDATION task.

This is NOT extraction. This is NOT interpretation. This is NOT analysis.

RULES:
- Use ONLY the provided input fields.
- Do NOT add new facts, numbers, or assumptions.
- Do NOT change page numbers.
- If the 'answer' is empty but 'evidence' exists, produce a concise factual summary.
- If both are empty, summary MUST be "Not disclosed in the report."

INPUT (JSON):
{json.dumps(record, indent=2)}

OUTPUT FORMAT (JSON ONLY):
{{
  "metric_id": "{record['metric_id']}",
  "summary": "<concise factual summary>",
  "key_facts": ["<fact 1>", "<fact 2>"],
  "page": "{record['page']}"
}}
"""

def consolidate_records(records: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    consolidated = []
    logger.info(f"Starting second-pass consolidation using model: {model}...")

    for rec in records:
        try:
            prompt = build_consolidation_prompt(rec)
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            # FIX: Clean potential Markdown wrapping before loading JSON
            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE)
            
            output = json.loads(content)
            consolidated.append(output)

        except Exception as e:
            logger.error(f"Consolidation failed for {rec.get('metric_id', 'Unknown')}: {str(e)}")
            continue

    return consolidated

# --------------------------------------------------
# Pipeline Function
# --------------------------------------------------

def run_post_processing_pipeline(
    input_file: Path,
    intermediate_file: Path,
    output_file: Path,
    model: str
):
    """
    Modular execution logic for parsing and consolidating ESG data.
    """
    try:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # 1. Parse Raw Text
        logger.info(f"Step 1: Parsing raw answers from {input_file}")
        raw_text = input_file.read_text(encoding="utf-8")
        extracted_records = parse_model_answers(raw_text)
        
        # Save intermediate
        intermediate_file.write_text(
            json.dumps(extracted_records, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # 2. Consolidate
        logger.info("Step 2: Consolidating records...")
        final_data = consolidate_records(extracted_records, model=model)

        # 3. Save Final
        output_file.write_text(
            json.dumps(final_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(f"Step 3: Processed {len(final_data)} metrics. Final output: {output_file}")

    except Exception as e:
        raise CustomException(e, sys)
