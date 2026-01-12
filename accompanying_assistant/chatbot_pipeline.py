import sys
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any
from litellm import completion

from utils.read_yaml import read_yaml
from utils.logger import logging
from utils.exception import CustomException
from retrieval_and_postprocessing.retrieval_full_pipeline import AdvancedRAGRetrievalEngine

logger = logging.getLogger(__name__)

class AccompanyingChatbot:
    def __init__(self, config_path: Path, db_path: Path):
        self.config = read_yaml(config_path)
        self.engine = AdvancedRAGRetrievalEngine(config_path=config_path, db_path=db_path)

    def _normalize_history(self, history: List[Any]) -> List[Dict[str, str]]:
        """
        Normalizes history into OpenAI-compatible format: <checks if format is correct and converts if possible>
        [{"role": "...", "content": "..."}]
        """
        normalized = []

        for item in history:
            try:
                # Already in correct format
                if isinstance(item, dict) and "role" in item and "content" in item:
                    normalized.append(item)

                # Gradio-style: ["user msg", "assistant msg"]
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    user_msg, assistant_msg = item
                    if user_msg:
                        normalized.append({"role": "user", "content": str(user_msg)})
                    if assistant_msg:
                        normalized.append({"role": "assistant", "content": str(assistant_msg)})

            except Exception as e:
                logger.warning(f"Dropped malformed history item: {item} | Error: {e}")

        return normalized


    def _generate_summary(self, history: List[Dict[str, str]]) -> str:
        """Compresses old history to save tokens while retaining context."""
        try:
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            prompt = self.config.prompts.summary_template.format(history=history_text)
            
            response = completion(
                model=self.config.memory.summary_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception:
            return "Summary unavailable."

    def get_response(self, question: str, history: List[Dict[str, str]] | None = None):
        try:
            if history is None:
                history = []
            else:
                history = self._normalize_history(history)


            summary = "None."

            if len(history) > self.config.memory.summarize_threshold:
                split_idx = -self.config.memory.max_history_messages
                summary = self._generate_summary(history[:split_idx])
                managed_history = history[split_idx:]
            else:
                managed_history = history

            context_chunks, _ = self.engine.get_context_advanced(question)

            if not context_chunks:
                context_text = "No relevant context found in the report."
            else:
                context_text = "\n\n".join(
                    f"[Page {c.metadata.get('page', 'unknown')}]: {c.page_content}"
                    for c in context_chunks
                )

            system_prompt = self.config.prompts.system_template.format(
                context=context_text,
                summary=summary
            )

            messages = (
                [{"role": "system", "content": system_prompt}]
                + managed_history
                + [{"role": "user", "content": question}]
            )

            response = completion(
                model=self.config.model.name,
                messages=messages,
                temperature=self.config.model.temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            raise CustomException(e, sys)



# --- GRADIO TEST INTERFACE ---
if __name__ == "__main__":
    # Settings for local debugging
    DEBUG_RUN_ID = "Infosys BRSR 2024_20260111_202132" # Replace with your active run_id
    DB_PATH = Path("runs") / DEBUG_RUN_ID / "chroma_db"
    CONFIG = Path("config/accompanying_chatbot_config.yaml")

    bot = AccompanyingChatbot(config_path=CONFIG, db_path=DB_PATH)

    def gradio_wrapper(message, history):
        """Adapts Gradio's history format to the bot's expected OpenAI format."""
        # Gradio 'messages' type already provides list[dict] with role/content
        return bot.get_response(message, history)

    demo = gr.ChatInterface(
        fn=gradio_wrapper,
        title="ESG Audit Assistant - Debugger",
        description=f"Testing Run ID: {DEBUG_RUN_ID}"
        )


    print(f"Launching Gradio test server for Run: {DEBUG_RUN_ID}...")
    demo.launch(server_name="127.0.0.1", server_port=7861)