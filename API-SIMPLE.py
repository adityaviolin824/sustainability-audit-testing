from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import shutil
import sys

# Core Pipeline Imports
from vectorstore_ingestion.full_ingestion_pipeline import run_ingestion_pipeline
from retrieval_and_postprocessing.retrieval_full_pipeline import AdvancedRAGRetrievalEngine
from qa_and_report_generation.report_generation_pipeline import ESGReportPipeline
# Import custom exception for the error handling if needed
from utils.exception import CustomException

app = FastAPI(title="ESG Audit API")

# --- SCHEMAS ---
class QueryRequest(BaseModel):
    run_id: str
    query: str

# --- ENDPOINTS ---

@app.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...) 
):
    """
    1. Ingestion: Saves PDF and triggers the embedding pipeline in the background.
    """
    base_dir = Path("runs")
    config = Path("config/ingestion_master_config.yaml")
    
    # 1. Setup Run Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(file.filename).stem}_{timestamp}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Persist Uploaded File
    pdf_path = run_dir / file.filename
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. Define Internal Paths
    paths = {
        "formatted_txt": run_dir / "formatted_report.txt",
        "chunks_debug": run_dir / "chunks_preview.txt",
        "db_path": run_dir / "chroma_db"
    }

    # 4. START BACKGROUND TASK (Uncommented and Active)
    # This returns the 'run_id' to the user immediately while processing continues
    background_tasks.add_task(run_ingestion_pipeline, pdf_path, config, paths)
    
    return {
        "status": "Processing Started",
        "run_id": run_name,
        "folder_path": str(run_dir)
    }


@app.post("/retrieve")
async def retrieve_context(request: QueryRequest):
    """
    2. Retrieval: Instant semantic search for specific auditor queries.
    """
    run_dir = Path("runs") / request.run_id
    db_path = run_dir / "chroma_db"
    config_path = Path("config/retrieval_master_config.yaml")

    engine = AdvancedRAGRetrievalEngine(config_path=config_path, db_path=db_path)
    context, used_query = engine.get_context_advanced(request.query)

    return {
        "expanded_query": used_query, 
        "results": [
            {
                "page": c.metadata.get("page"),
                "content": c.page_content,
                "principle": c.metadata.get("principle")
            } for c in context
        ]
    }


@app.post("/generate-report/{run_id}")
async def generate_report(run_id: str, background_tasks: BackgroundTasks):
    """
    3. Report Generation: Triggers the heavy LLM batch extraction. 
    Also moved to background to prevent timeout.
    """
    session_run_dir = Path("runs") / run_id
    master_config = Path("config/qa_and_report_master_config.yaml")

    # We wrap the pipeline run in a simple function for the background worker
    def run_pipeline():
        try:
            pipeline = ESGReportPipeline(master_config_path=master_config, run_dir=session_run_dir)
            pipeline.run()
        except Exception as e:
            print(f"Background Report Task Failed: {e}")

    background_tasks.add_task(run_pipeline)
    
    return {
        "status": "Report generation triggered",
        "run_id": run_id,
        "note": "Check the run directory for files once complete."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)