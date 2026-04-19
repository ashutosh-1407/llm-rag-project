# from sentence_transformers import SentenceTransformer
from src.utils.constants import CACHE_PATH
from src.utils.helper import logger
from src.rag.embedder import embed_text
import faiss
import os
import pickle


# model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

class Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        logger.info("Loading/saving the embeddings")
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            # embeddings = model.encode(chunks)
            self.embeddings = embed_text(chunks)
            os.makedirs("cache", exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(self.embeddings, f)
        logger.info("Building FAISS")                
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
    
    def retrieve(self, query, k=3):
        logger.info(f"Retrieving the context | query={query}")
        # q_embedding = model.encode([query])
        q_embedding = embed_text(query)
        _, indices = self.index.search(q_embedding, k)
        context = [self.chunks[i] for i in indices[0]]
        logger.info(f"Retrieved context for the query: {query} is {context}")
        return context
