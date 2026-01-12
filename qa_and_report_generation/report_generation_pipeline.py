import sys
from pathlib import Path

# Module Imports
from qa_and_report_generation.retrieval_and_qa import run_esg_batch_extraction
from qa_and_report_generation.responses_postprocessing import run_post_processing_pipeline
from qa_and_report_generation.report_formatting import run_reporting_pipeline

# Utility Imports
from utils.logger import logging
from utils.exception import CustomException
from utils.read_yaml import read_yaml

logger = logging.getLogger(__name__)

class ESGReportPipeline:
    def __init__(self, master_config_path: Path, run_dir: Path):
        """
        Initializes the pipeline with a dynamic run directory.
        """
        try:
            self.master_config = read_yaml(master_config_path)
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Pipeline initialized for directory: {self.run_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        """Sequential execution of the ESG extraction and reporting stages."""
        try:
            # 1. Extraction (retrieval_and_qa.py)
            logger.info(">>> Stage 1: Batch Extraction <<<")
            run_esg_batch_extraction(
                config_path=self.master_config.pipeline.retrieval_config,
                db_path=str(self.run_dir / "chroma_db"),
                question_path=self.master_config.pipeline.question_path,
                run_dir=self.run_dir,
                model=self.master_config.models.extraction_model
            )

            # 2. Post-Processing (responses_postprocessing.py)
            logger.info(">>> Stage 2: Post-Processing & Consolidation <<<")
            run_post_processing_pipeline(
                input_file=self.run_dir / self.master_config.filenames.raw_responses_txt,
                intermediate_file=self.run_dir / self.master_config.filenames.intermediate_json,
                output_file=self.run_dir / self.master_config.filenames.consolidated_json,
                model=self.master_config.models.refinement_model
            )

            # 3. Formatting (report_formatting.py)
            logger.info(">>> Stage 3: Report Formatting <<<")
            run_reporting_pipeline(
                consolidated_json_path=self.run_dir / self.master_config.filenames.consolidated_json,
                questions_jsonl_path=Path(self.master_config.pipeline.question_path),
                output_docx_path=self.run_dir / self.master_config.filenames.final_docx,
                output_xlsx_path=self.run_dir / self.master_config.filenames.final_xlsx
            )

            logger.info(f"✅ Pipeline successful for {self.run_dir.name}")

        except CustomException as ce:
            logger.error(f"Pipeline failed at stage: {ce}")
            raise ce
        except Exception as e:
            logger.error(f"System Error: {e}")
            raise CustomException(e, sys)
