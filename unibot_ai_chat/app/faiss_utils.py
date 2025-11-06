import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Modellerin ve dosyaların yolları
INDEX_PATH = "data/faiss_index.index"
META_PATH = "data/faiss_metadata.pkl"

# Model ve index yükleme
model = SentenceTransformer("intfloat/multilingual-e5-base")

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

def search_similar_chunks(query, top_k=5, score_threshold=0.65):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(len(indices[0])):
        if distances[0][i] <= score_threshold:
            chunk_meta = metadata[indices[0][i]]
            text = chunk_meta.get("text", "")  # text metadata'ya eklendiyse
            if not text:
                # Geriye dönük uyumluluk için
                text = chunk_meta.get("content", "")
            results.append(text)
    return results

