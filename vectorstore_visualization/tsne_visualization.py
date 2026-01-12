import sys
import random
import re
import numpy as np
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go
from sklearn.manifold import TSNE
from chromadb import PersistentClient

from utils.logger import logging
from utils.exception import CustomException

logger = logging.getLogger(__name__)


class BRSRVectorVisualizer:
    def __init__(self, db_path: Path):
        """Initializes the visualizer and connects to the first available collection."""
        try:
            if not db_path.exists():
                raise FileNotFoundError(f"Vector database not found at {db_path}")

            self.chroma = PersistentClient(path=str(db_path))
            collections = self.chroma.list_collections()

            if not collections:
                raise ValueError("No collections found in the specified path.")

            self.collection_name = collections[0].name
            self.collection = self.chroma.get_collection(self.collection_name)

            logger.info(f"Connected to collection: {self.collection_name}")

        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_principle_color(self, meta: Dict) -> str:
        principle_colors = {
            "Principle 1": "#1f77b4",
            "Principle 2": "#ff7f0e",
            "Principle 3": "#2ca02c",
            "Principle 6": "#d62728",
            "Principle 8": "#9467bd",
            "General Information": "#7f7f7f",
        }

        raw = " ".join(
            str(v) for v in [
                meta.get("principle"),
                meta.get("principle_context"),
                meta.get("section"),
                meta.get("heading"),
            ]
            if v
        )

        if not raw:
            return "#7f7f7f"

        for key, color in principle_colors.items():
            if key in raw:
                return color

        return "#bcbd22"

    def _is_narrative_chunk(self, meta: Dict) -> bool:
        """Strict filter: only narrative chunks are allowed."""
        return meta.get("type") == "narrative"

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """
        Cleans PDF-extracted text for display:
        - removes tabs and newlines
        - collapses multiple spaces
        - trims edges
        """
        if not text:
            return ""

        text = text.replace("\t", " ").replace("\n", " ")
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Main visualization
    # ------------------------------------------------------------------
    def run_visualization(
        self,
        max_points: int,
        perplexity: int,
        preview_len: int,
        n_components: int = 2,
    ):
        try:
            if n_components not in (2, 3):
                raise ValueError("n_components must be 2 or 3")

            # 1. Load data
            result = self.collection.get(include=["embeddings", "documents", "metadatas"])
            embeddings = result["embeddings"]
            documents = result["documents"]
            metadatas = result["metadatas"]

            # 2. Filter narrative chunks only
            filtered = [
                (e, d, m)
                for e, d, m in zip(embeddings, documents, metadatas)
                if self._is_narrative_chunk(m)
            ]

            if not filtered:
                raise ValueError("No narrative chunks found for visualization.")

            embeddings, documents, metadatas = zip(*filtered)
            embeddings = list(embeddings)
            documents = list(documents)
            metadatas = list(metadatas)

            total_points = len(embeddings)
            logger.info(f"Narrative vectors in DB: {total_points}")

            # 3. Sampling
            if total_points > max_points:
                idx = random.sample(range(total_points), max_points)
                vectors = np.array([embeddings[i] for i in idx])
                subset_docs = [documents[i] for i in idx]
                subset_meta = [metadatas[i] for i in idx]
            else:
                vectors = np.array(embeddings)
                subset_docs = documents
                subset_meta = metadatas

            # 4. Colors
            colors = [self.get_principle_color(m) for m in subset_meta]

            # 5. Dimensionality reduction
            logger.info(f"Computing t-SNE ({n_components}D) for {len(vectors)} vectors...")
            tsne = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(vectors) - 1),
                learning_rate=200,
                init="pca",
                random_state=42,
            )
            reduced = tsne.fit_transform(vectors)

            # 6. Hover text (cleaned)
            hover_text = []
            for doc, meta in zip(subset_docs, subset_meta):
                clean_text = self._clean_text(doc or "")
                preview = clean_text[:preview_len] + (
                    "…" if len(clean_text) > preview_len else ""
                )

                hover_text.append(
                    f"{preview}<br>"
                    f"<hr>"
                    f"Principle: {meta.get('principle', 'N/A')}<br>"
                    f"Page: {meta.get('page', 'N/A')} | Source: {meta.get('source', 'N/A')}"
                )

            # 7. Plot
            if n_components == 2:
                trace = go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=1, color="white"),
                    ),
                    hoverinfo="text",
                    text=hover_text,
                )
            else:
                trace = go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(size=4, color=colors, opacity=0.8),
                    hoverinfo="text",
                    text=hover_text,
                )

            fig = go.Figure(trace)

            fig.update_layout(
                title=f"Semantic Map (Narrative Only): {self.collection_name}",
                template="plotly_white",
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )

            if n_components == 2:
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
                fig.update_yaxes(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    scaleanchor="x",
                    scaleratio=1,
                )

            return fig.to_json()

        except Exception as e:
            raise CustomException(e, sys)
