import json
import re
import sys
from typing import List, Dict, Tuple
from pathlib import Path

from utils.logger import logging
from utils.exception import CustomException

# Initialize Logger
logger = logging.getLogger(__name__)

# =========================================================
# SEMANTIC PRINCIPLE MAPPING (Enables better Vector Matching)
# =========================================================

PRINCIPLE_MAP = {
    "1": "Ethics, Transparency and Accountability - Businesses should conduct and govern themselves with integrity and in a manner that is ethical, transparent and accountable.",
    "2": "Sustainable and Safe Goods and Services - Businesses should provide goods and services in a manner that is sustainable and safe.",
    "3": "Well-being of all Employees - Businesses should respect and promote the well-being of all employees, including those in their value chains.",
    "4": "Stakeholder Responsiveness - Businesses should respect the interests of and be responsive to all its stakeholders.",
    "5": "Human Rights - Businesses should respect and promote human rights.",
    "6": "Environmental Protection - Businesses should respect and make efforts to protect and restore the environment.",
    "7": "Responsible Advocacy - Businesses, when engaging in influencing public and regulatory policy, should do so in a manner that is responsible and transparent.",
    "8": "Inclusive Growth - Businesses should promote inclusive growth and equitable development.",
    "9": "Consumer Value - Businesses should engage with and provide value to their consumers in a responsible manner."
}

# =========================================================
# CLEANING & HELPERS
# =========================================================

def clean_raw_text(text: str) -> str:
    """Remove extraction metadata and separators but KEEP page breaks."""
    text = re.sub(r"--- METADATA START ---.*?--- METADATA END ---", "", text, flags=re.DOTALL)
    text = re.sub(r"={30,}", "", text)
    return text.strip()

def looks_like_table_row(line: str) -> bool:
    """Heuristic to check if a line represents a table row based on spacing/digits."""
    return len(re.findall(r"\s{3,}", line)) >= 1 or bool(re.search(r"\d", line))

# =========================================================
# PASS 1: HIGH-PRECISION TABLE EXTRACTION
# =========================================================

TABLE_HEADER_KEYWORDS = ["sr.", "total", "male", "female", "%", "category", "fy", "particulars", "no.", "unit", "amount"]

def extract_tables_precise(lines: List[str]) -> Tuple[List[Dict], List[bool]]:
    """Identifies tables while tracking Page and Full Principle Name."""
    used_mask = [False] * len(lines)
    chunks = []
    current_page = 1
    current_princ_text = "General Information"
    
    i = 0
    table_count = 0
    while i < len(lines):
        line = lines[i]
        
        # State Tracking
        if "<<<" in line: current_page += 1
        princ_match = re.search(r"PRINCIPLE\s+(\d+)", line, re.IGNORECASE)
        if princ_match:
            pid = princ_match.group(1)
            current_princ_text = f"Principle {pid}: {PRINCIPLE_MAP.get(pid, 'Unknown')}"

        # Detection Heuristics
        window = lines[i:i + 8]
        header_hits = sum(any(k in l.lower() for k in TABLE_HEADER_KEYWORDS) for l in window)
        spaced_lines = sum(len(re.findall(r"\s{3,}", l)) >= 2 for l in window)
        numeric_lines = sum(bool(re.search(r"\d", l)) for l in window)

        if header_hits >= 2 and spaced_lines >= 2 and numeric_lines >= 2:
            start = i
            j = i
            non_table_streak = 0
            while j < len(lines):
                curr_line = lines[j].strip()
                if not curr_line: j += 1; continue
                
                # Check for BRSR Section breaks
                if (re.match(r"^note:", curr_line.lower()) or 
                    re.match(r"^\s*(section\s+[a-z]:|principle\s+\d+)", curr_line.lower()) or 
                    re.match(r"^\s*(essential indicators|leadership indicators)", curr_line.lower())):
                    break
                
                if not looks_like_table_row(curr_line): non_table_streak += 1
                else: non_table_streak = 0
                
                if non_table_streak >= 3: break
                j += 1

            table_text = "\n".join(lines[start:j]).strip()
            if len(table_text.splitlines()) >= 4:
                chunks.append({
                    "chunk_id": f"table_{table_count}",
                    "chunk_type": "table",
                    "page_number": current_page,
                    "principle_context": current_princ_text,
                    "metadata": {
                        "type": "table",
                        "page": current_page,
                        "principle": current_princ_text
                    },
                    "text": f"[CONTEXT | PAGE: {current_page} | {current_princ_text}]\n\n{table_text}"
                })
                table_count += 1
                for k in range(start, j): used_mask[k] = True
                i = j
            else: i += 1
        else: i += 1
    return chunks, used_mask

# =========================================================
# PASS 2: NARRATIVE CHUNKING
# =========================================================

QUESTION_PATTERN = re.compile(r"^\s*(\d+\.\s+|Section\s+[A-Z]:|Principle\s+\d+|[IVX]+\.)", re.MULTILINE)
MAX_CHUNK_LINES = 25
OVERLAP_LINES = 5

def extract_narrative_chunks(lines: List[str], source_file: str) -> List[Dict]:
    chunks = []
    buffer = []
    chunk_id = 0
    current_page = 1
    current_princ_text = "General Information"

    def flush(page, princ_name):
        nonlocal chunk_id
        if buffer:
            chunk_text = "\n".join(buffer).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"narrative_{chunk_id}",
                    "chunk_type": "narrative",
                    "page_number": page,
                    "principle_context": princ_name,
                    "metadata": {
                        "source": source_file,
                        "type": "narrative",
                        "page": page,
                        "principle": princ_name
                    },
                    "text": f"[CONTEXT | PAGE: {page} | {princ_name}]\n\n{chunk_text}"
                })
                chunk_id += 1

    for line in lines:
        if "<<<" in line: current_page += 1
        
        # --- FIX 1: Change match to search to catch headers buried in noise ---
        princ_match = re.search(r"PRINCIPLE\s+(\d+)", line, re.IGNORECASE)
        
        if princ_match:
            # --- FIX 2: Flush the OLD context before updating to the NEW one ---
            if buffer:
                flush(current_page, current_princ_text)
                buffer = [] # Start fresh for the new Principle
            
            pid = princ_match.group(1)
            current_princ_text = f"Principle {pid}: {PRINCIPLE_MAP.get(pid, 'Unknown')}"

        # Existing Question Pattern logic
        if QUESTION_PATTERN.match(line) and buffer:
            flush(current_page, current_princ_text)
            buffer[:] = buffer[-OVERLAP_LINES:]

        buffer.append(line)
        
        # Existing Max Lines logic
        if len(buffer) >= MAX_CHUNK_LINES:
            flush(current_page, current_princ_text)
            buffer[:] = buffer[-OVERLAP_LINES:]

    flush(current_page, current_princ_text)
    return chunks

# =========================================================
# FINAL PIPELINE & INTEGRATION
# =========================================================

def chunk_document_final(raw_text: str, source_file: str) -> List[Dict]:
    """Orchestrates cleaning, table extraction, and narrative chunking."""
    try:
        clean_text = clean_raw_text(raw_text)
        lines = clean_text.splitlines()

        # Extract Tables first to "mask" them from narrative chunking
        table_chunks, _ = extract_tables_precise(lines)
        for t in table_chunks: 
            t["metadata"]["source"] = source_file
            t["source_file"] = source_file

        # Extract Narrative (Text) portions
        narrative_chunks = extract_narrative_chunks(lines, source_file)
        for n in narrative_chunks:
            n["source_file"] = source_file

        return table_chunks + narrative_chunks
    except Exception as e:
        raise CustomException(e, sys)

def run_chunking_standalone(input_path: Path, output_path: Path, source_name: str):
    """Standalone execution logic using pathlib."""
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        full_text = input_path.read_text(encoding="utf-8")
        chunks = chunk_document_final(full_text, source_name)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w", encoding="utf-8") as f:
            for ch in chunks:
                f.write("=== CHUNK START ===\n")
                meta = {k: v for k, v in ch.items() if k != "text"}
                f.write(json.dumps(meta, indent=2))
                f.write("\n--- TEXT ---\n")
                f.write(ch["text"])
                f.write("\n=== CHUNK END ===\n\n")

        logger.info(f"[OK] Generated {len(chunks)} contextualized chunks in {output_path}")
        return chunks

    except Exception as e:
        raise CustomException(e, sys)



if __name__ == "__main__":
    # Example paths for standalone testing
    BASE = Path("documents_and_vectorstore")
    INPUT = BASE / "formatted_report.txt"
    OUTPUT = Path("testing") / "demo_chunking.txt"
    
    run_chunking_standalone(INPUT, OUTPUT, "Jio_Financial_BRSR_2024.pdf")