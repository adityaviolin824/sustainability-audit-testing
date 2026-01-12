import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from utils.read_yaml import read_yaml
from utils.logger import logging
from utils.security_gate import SecurityGate

# --- 1. GLOBAL INITIALIZATION ---
# Initialize the shield once at the top level
BLACKLIST = ["ignore previous instructions", "system prompt", "developer mode", "print context"]
SHIELD = SecurityGate(blacklist=BLACKLIST, threshold=0.85)

logger = logging.getLogger(__name__)

# Core Pipeline Logic
from vectorstore_ingestion.full_ingestion_pipeline import run_ingestion_pipeline
from qa_and_report_generation.report_generation_pipeline import ESGReportPipeline
# from vectorstore_visualization.tsne_visualization import BRSRVectorVisualizer # too heavy for 512mb ram
from vectorstore_visualization.pca_visualization import BRSRVectorVisualizer
from accompanying_assistant.chatbot_pipeline import AccompanyingChatbot

app = FastAPI(title="ESG Audit API")

# --- 1. CONFIGURATION & STATE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tracks progress of long-running background tasks
TASK_STATE: Dict[str, str] = {}

# --- 2. MAINTENANCE ---
def run_filesystem_cleanup(base_dir: str = "runs", max_age_hours: int = 4):
    """Cleans up processed 'run' folders to manage disk space."""
    base_path = Path(base_dir)
    if not base_path.exists(): return
    now = time.time()
    for folder in base_path.iterdir():
        if folder.is_dir() and (now - folder.stat().st_ctime) > (max_age_hours * 3600):
            run_id = folder.name
            # Only delete if not currently processing
            if run_id not in TASK_STATE or TASK_STATE[run_id] in ["ready", "completed", "failed"]:
                shutil.rmtree(folder, ignore_errors=True)
                TASK_STATE.pop(run_id, None)

# --- 3. AUDIT WORKFLOW ENDPOINTS ---

@app.post("/audit/ingest")
async def start_document_ingestion(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    STAGE 1: Ingestion
    Fully dynamic: Resolves all paths directly from the ConfigBox.
    """
    run_filesystem_cleanup()
    
    # 1. Load Config
    config_path = Path("config/ingestion_master_config.yaml")
    config = read_yaml(config_path)
    
    # 2. Setup Directories
    base_storage = Path(config.storage.base_run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{Path(file.filename).stem}_{timestamp}"
    run_dir = base_storage / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Save PDF
    pdf_path = run_dir / file.filename
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 4. DYNAMIC PATH RESOLUTION
    # This loop takes every key in your YAML 'output_paths' and joins it to run_dir
    # No more hardcoded dictionary!
    output_paths = {
        key: run_dir / filename 
        for key, filename in config.output_paths.items()
    }

    def execute_ingestion():
        TASK_STATE[run_name] = "ingesting"
        try:
            run_ingestion_pipeline(pdf_path, config_path, output_paths)
            TASK_STATE[run_name] = "ready"
        except Exception as e:
            print(f"Error: {e}")
            TASK_STATE[run_name] = "failed_ingestion"

    background_tasks.add_task(execute_ingestion)
    return {"run_id": run_name, "status": "ingesting"}



@app.post("/audit/generate-report/{run_id}")
async def start_report_generation(run_id: str, background_tasks: BackgroundTasks):
    """
    STAGE 2 & 3: Retrieval, QA, and Formatting
    Triggers the batch questioning of the document and generates the final ESG report.
    """
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        return {"error": "Run ID not found. Please ingest the document first."}

    def execute_report_pipeline():
        TASK_STATE[run_id] = "generating_report"
        try:
            # ESGReportPipeline handles Batch Retrieval -> LLM QA -> Report Formatting [retrieval config path inside this config]
            master_config = Path("config/qa_and_report_master_config.yaml")
            pipeline = ESGReportPipeline(master_config_path=master_config, run_dir=run_dir)
            pipeline.run()
            TASK_STATE[run_id] = "completed"
        except Exception:
            TASK_STATE[run_id] = "failed_report"

    background_tasks.add_task(execute_report_pipeline)
    return {"run_id": run_id, "status": "generating_report"}



@app.post("/audit/chat/{run_id}")
async def chat_with_report(run_id: str, payload: Dict):
    """
    Chat with an already-ingested ESG report.

    Expected payload:
    {
        "question": "What are the Scope 3 emissions?",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    run_dir = Path("runs") / run_id
    db_path = run_dir / "chroma_db"

    if not db_path.exists():
        return {"run_id": run_id, "error": "Vectorstore not found."}

    # --- Query Injection Guard ---
    is_evil, reason = SHIELD.is_malicious(payload["question"])
    if is_evil:
        logger.warning(f"BLOCKED QUERY: {run_id} | Reason: {reason}")
        return {
            "run_id": run_id,
            "error": f"Security Violation: {reason}"
        }

    try:
        config_path = Path("config/accompanying_chatbot_config.yaml")
        bot = AccompanyingChatbot(config_path=config_path, db_path=db_path)

        answer = bot.get_response(
            question=payload["question"],
            history=payload.get("history", [])
        )

        return {
            "run_id": run_id,
            "answer": answer
        }

    except Exception:
        logger.exception("Chat endpoint failed")
        return {
            "run_id": run_id,
            "error": "Chat processing failed"
        }
    

@app.get("/audit/download/{run_id}/{file_type}")
async def download_audit_file(run_id: str, file_type: str):
    """
    Downloads specific audit files:
    - docx: ESG_Audit_Document.docx
    - xlsx: ESG_Disclosures.xlsx
    - md:   batchwise_responses_audit.md
    """
    file_map = {
        "docx": "ESG_Audit_Document.docx",
        "xlsx": "ESG_Disclosures.xlsx",
        "md": "batchwise_responses_audit.md"
    }

    if file_type not in file_map:
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid file type. Choose docx, xlsx, or md."}
        )

    file_path = Path("runs") / run_id / file_map[file_type]

    if not file_path.exists():
        logger.error(f"Download failed: {file_path} not found.")
        return JSONResponse(
            status_code=404, 
            content={"error": "File not found. Ensure report generation is complete."}
        )

    # Define media types for better browser handling
    media_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "md": "text/markdown"
    }

    return FileResponse(
        path=file_path,
        filename=file_map[file_type],
        media_type=media_types[file_type]
    )



# @app.get("/audit/visualize/{run_id}") # TSNE VERSION
# async def get_vector_visualization(
#     run_id: str,
#     n_components: int = Query(
#         2,
#         ge=2,
#         le=3,
#         description="Dimensionality of t-SNE projection (2 or 3)",
#     ),
# ):
#     """
#     Generates a t-SNE semantic map of the vectorstore for a specific run.

#     Optional query params:
#     - n_components: 2 (default) or 3
#     """
#     run_dir = Path("runs") / run_id
#     db_path = run_dir / "chroma_db"

#     if not db_path.exists():
#         return {"error": "Vectorstore not found for visualization."}

#     try:
#         # Load visualization params
#         config = read_yaml(Path("config/ingestion_master_config.yaml"))
#         params = config.visualization_params

#         visualizer = BRSRVectorVisualizer(db_path=db_path)

#         fig_json = visualizer.run_visualization(
#             max_points=params.max_points,
#             perplexity=params.tsne_perplexity,
#             preview_len=params.text_preview_len,
#             n_components=n_components,
#         )

#         return {
#             "run_id": run_id,
#             "n_components": n_components,
#             "plot_json": fig_json,
#         }

#     except Exception as e:
#         logger.error(f"Visualization endpoint failed for run_id={run_id}: {e}")
#         return {"error": "Failed to generate visualization"}

# PCA VERSION
@app.get("/audit/visualize/{run_id}")
async def get_vector_visualization(
    run_id: str,
    n_components: int = Query(
        2,
        ge=2,
        le=3,
        description="Dimensionality of PCA projection (2 or 3)", # Updated description
    ),
):
    """
    Generates a PCA semantic map of the vectorstore for a specific run.
    """
    run_dir = Path("runs") / run_id
    db_path = run_dir / "chroma_db"

    if not db_path.exists():
        return {"error": "Vectorstore not found for visualization."}

    try:
        # Load visualization params
        config = read_yaml(Path("config/ingestion_master_config.yaml"))
        params = config.visualization_params

        visualizer = BRSRVectorVisualizer(db_path=db_path)

        # Removed 'perplexity' argument as it's no longer in the method signature
        fig_json = visualizer.run_visualization(
            max_points=params.max_points,
            preview_len=params.text_preview_len,
            n_components=n_components,
        )

        return {
            "run_id": run_id,
            "n_components": n_components,
            "plot_json": fig_json,
        }

    except Exception as e:
        logger.error(f"Visualization endpoint failed for run_id={run_id}: {e}")
        return {"error": str(e)} # Returning the error string helps debugging
    

@app.get("/audit/status/{run_id}")
async def fetch_audit_status(run_id: str):
    """Monitors the progress of the current ingestion or generation task."""
    return {"run_id": run_id, "status": TASK_STATE.get(run_id, "not_found")}


@app.get("/status")
async def health_check():
    # You can add logic here to check if API keys are loaded
    return JSONResponse(
        content={"status": "healthy", "version": "1.0.0"},
        status_code=200
    )

