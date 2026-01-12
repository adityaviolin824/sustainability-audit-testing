from chromadb import PersistentClient
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import random

DB_NAME = r"vector_db\sustainability_index"
COLLECTION_NAME = "esg_audit_demo"

# ---- CONFIG (512 MB RAM SAFE) ----
MAX_POINTS = 500
TSNE_PERPLEXITY = 30
TEXT_PREVIEW_LEN = 120   # smaller to avoid giant hover boxes

## LOAD VECTORSTORE
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

result = collection.get(include=["embeddings", "documents", "metadatas"])

embeddings = result["embeddings"]
documents = result["documents"]
metadatas = result["metadatas"]

total_points = len(embeddings)
print(f"Total vectors in DB: {total_points}")

# ---- SAMPLE ----
if total_points > MAX_POINTS:
    idx = random.sample(range(total_points), MAX_POINTS)
    vectors = np.array([embeddings[i] for i in idx])
    documents = [documents[i] for i in idx]
    metadatas = [metadatas[i] for i in idx]
else:
    vectors = np.array(embeddings)

print(f"Visualizing {len(vectors)} vectors")

# ---- COLOR MAPPING ----
def normalize_type(t):
    if not t:
        return "unknown"
    if "table" in t:
        return "tabular"
    if "narrative" in t:
        return "narrative"
    return "other"

color_map = {
    "tabular": "red",
    "narrative": "blue",
    "policy": "green",
    "other": "gray",
    "unknown": "gray",
}

doc_types = [normalize_type(m.get("source_type")) for m in metadatas]
colors = [color_map[t] for t in doc_types]

# ---- t-SNE ----
tsne = TSNE(
    n_components=2,
    perplexity=min(TSNE_PERPLEXITY, len(vectors) - 1),
    learning_rate=200,
    random_state=42,
)

reduced_vectors = tsne.fit_transform(vectors)

# ---- COMPACT HOVER TEXT ----
hover_text = [
    f"""
    <b>{m.get('primary_topic')}</b><br>
    Pillar: {m.get('audit_pillar')} | Page: {m.get('page_number')}<br>
    Type: {m.get('source_type')}<br><br>
    {d[:TEXT_PREVIEW_LEN].replace("\\n", " ")}...
    """
    for d, m in zip(documents, metadatas)
]

# ---- PLOT ----
fig = go.Figure(
    go.Scatter(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        mode="markers",
        marker=dict(size=5, color=colors, opacity=0.7),
        hoverinfo="text",
        text=hover_text,
    )
)

fig.update_layout(
    title="Semantic Map of Vectorstore (t-SNE)",
    width=720,     # smaller, notebook-friendly
    height=520,
    margin=dict(r=10, b=10, l=10, t=40),
)

fig.show()
