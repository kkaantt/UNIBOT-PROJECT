import json
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("intfloat/multilingual-e5-base") # hem ingilizce hem Türkçe için uygun
DATA_PATH = "data/tum_bolumler_chunks_enhanced.jsonl" #Bölünmüş veri dosyası
INDEX_PATH = "data/faiss_index.index"
META_PATH = "data/faiss_metadata.pkl"

texts = []
metadatas = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        texts.append(item["text"])
        metadatas.append({
            "text": item["text"],
            "metadata": item["metadata"] })

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
with open(META_PATH, "wb") as f:
    pickle.dump(metadatas, f)

print("✅ FAISS index oluşturuldu ve kaydedildi.")
