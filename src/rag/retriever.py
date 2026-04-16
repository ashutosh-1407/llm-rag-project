from sentence_transformers import SentenceTransformer
from src.utils.constants import SENTENCE_TRANSFORMER_MODEL_NAME, CACHE_PATH
import faiss
import numpy as np
import os
import pickle


model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            embeddings = model.encode(chunks)
            self.embeddings = np.array(embeddings).astype("float32")
            os.makedirs("cache", exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(self.embeddings, f)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
    
    def retrieve(self, query, k=3):
        q_embedding = model.encode([query])
        q_embedding = np.array(q_embedding).astype("float32")
        _, indices = self.index.search(q_embedding, k)
        return [self.chunks[i] for i in indices[0]]
