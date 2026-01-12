import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from unstract.llmwhisperer import LLMWhispererClientV2
from pypdf import PdfReader

from utils.logger import logging
from utils.exception import CustomException

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class ReportIntakePipeline:
    def __init__(self, hard_page_limit: int = 100):
        """
        Initializes the intake pipeline with a safety net for page limits.
        Loads API keys for load balancing between multiple accounts.
        """
        self.hard_page_limit = hard_page_limit
        
        self.api_keys = [os.getenv(f"LLMWHISPERER_API_KEY_{i}") for i in range(1, 6)]
        self.api_keys = [k for k in self.api_keys if k]
        
        if not self.api_keys:
            logger.warning("No LLMWhisperer API keys detected in environment variables.")

    def get_pdf_page_count(self, file_path: Path) -> int:
        """Locally check the PDF page count."""
        try:
            with file_path.open("rb") as f:
                reader = PdfReader(f)
                return len(reader.pages)
        except Exception as e:
            logger.error(f"Could not read PDF page count: {e}")
            return 0

    def process_chunk(
        self,
        file_path: Path,
        pages: str,
        key_index: int
    ) -> Optional[Dict[str, Any]]:
        """Handles API calls to LLMWhisperer for specific page batches."""
        api_key = self.api_keys[key_index % len(self.api_keys)]
        try:
            client = LLMWhispererClientV2(api_key=api_key)
            result = client.whisper(
                file_path=str(file_path),
                pages_to_extract=pages,
                mode="high_quality",
                wait_for_completion=True,
                wait_timeout=300
            )

            # CRITICAL: prevent aggressive socket reuse / polling collisions
            time.sleep(1.5)

            return result

        except Exception as ex:
            logger.error(f"API failure at pages {pages}: {ex}")
            return None

    def save_consolidated_report(
        self,
        results: List[Any],
        source_file: Path,
        out_path: Path
    ):
        """Consolidates extracted results into a single text report."""
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with out_path.open("w", encoding="utf-8") as f:
                for res in results:
                    if not res:
                        continue

                    if isinstance(res, list):
                        res = res[0]

                    extraction = res.get("extraction", {})
                    content = extraction.get("result_text") or res.get("result_text") or ""
                    line_metadata = extraction.get("line_metadata", [])
                    metadata_block = extraction.get("metadata", {})

                    y_coords = [
                        l[1] for l in line_metadata
                        if isinstance(l, (list, tuple)) and len(l) > 1 and l[1] > 0
                    ]

                    header = {
                        "source_file": source_file.name,
                        "pages_in_batch": list(metadata_block.keys()),
                        "bbox_y_range": [min(y_coords), max(y_coords)] if y_coords else [0, 0]
                    }

                    f.write(
                        "--- METADATA START ---\n"
                        + json.dumps(header, indent=4)
                        + "\n--- METADATA END ---\n\n"
                    )
                    f.write(content + "\n\n" + "=" * 60 + "\n\n")

        except Exception as e:
            raise CustomException(e, sys)

    def run_report_ingestion(
        self,
        file_path: str,
        output_txt: str,
        start_page: int = 1,
        batch_size: int = 5,
        max_workers: int = 2
    ):
        """Orchestrates extraction while respecting safety limits."""
        try:
            pdf_path = Path(file_path)
            out_path = Path(output_txt)

            actual_total_pages = self.get_pdf_page_count(pdf_path)
            effective_end_page = min(actual_total_pages, self.hard_page_limit)

            logger.info(
                f"[Safety Net] Actual pages: {actual_total_pages}. "
                f"Processing up to: {effective_end_page}"
            )

            if effective_end_page < start_page:
                logger.warning("Start page exceeds document length or hard limit.")
                return

            page_batches = []
            for i in range(start_page, effective_end_page + 1, batch_size):
                chunk_end = min(i + batch_size - 1, effective_end_page)
                page_batches.append(f"{i}-{chunk_end}")

            # FORCE SERIAL EXECUTION (minimal but essential)
            max_workers = 1

            logger.info(f"Dispatching {len(page_batches)} batches to LLMWhisperer...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_chunk, pdf_path, pages, idx)
                    for idx, pages in enumerate(page_batches)
                ]
                results = [f.result() for f in futures]

            self.save_consolidated_report(results, pdf_path, out_path)
            logger.info(f"Consolidated report saved to {out_path}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion_pipeline = ReportIntakePipeline(hard_page_limit=100)

    CONFIG = {
        "file_path": r"reference\sample_documents\Jio Financial Services BRSR 2024.pdf",
        "output_txt": "documents_and_vectorstore/formatted_report.txt",
        "start_page": 1,
        "batch_size": 5,
        "max_workers": 1
    }

    ingestion_pipeline.run_report_ingestion(**CONFIG)
