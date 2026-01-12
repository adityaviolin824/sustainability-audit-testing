from utils.logger import logging
from utils.exception import CustomException
from vectorstore_ingestion.reports_intake import run_chunk_intake_pipeline, export_demo_report
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_DIR = r"DATA\Official Reference Material"
    REPORT_PATH = r"testing\demo_chunking.txt"
    
    try:
        # --- EXECUTE PIPELINE ---
        final_list = run_chunk_intake_pipeline(
            folder_path=DATA_DIR,
            start_pg=1,
            end_pg=1000,
            chunk_sz=500,
            chunk_ov=100,
            LLM_ENRICHMENT = True #####
        )
        
        # --- SAVE AUDIT REPORT ---
        export_demo_report(final_list, REPORT_PATH)
        
        logger.info("### FULL SYSTEM SUCCESSFUL ###")
        
    except CustomException:
        # Already logged within the functions
        pass