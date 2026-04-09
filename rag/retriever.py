
from __future__ import annotations

import os
import pickle
import numpy as np
import faiss
import certifi
from typing import Optional

os.environ["SSL_CERT_FILE"] = certifi.where()

from sentence_transformers import SentenceTransformer, CrossEncoder

VECTOR_STORE_PATH = "rag/vector_store"
embedding_model   = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker          = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def load_vector_store():
    index = faiss.read_index(f"{VECTOR_STORE_PATH}/faiss_index")
    with open(f"{VECTOR_STORE_PATH}/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    return index, texts


index, texts = load_vector_store()
try:
    from rank_bm25 import BM25Okapi

    _tokenized = [t.lower().split() for t in texts]
    _bm25      = BM25Okapi(_tokenized)
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    _bm25 = None


def expand_query(query: str) -> list[str]:
    
    return [
        query,
        f"{query} explained simply",
        f"what is {query}",
        f"{query} step by step",
    ]


def _hyde_hypothetical_doc(query: str) -> Optional[str]:
    
    try:
        # Lazy import to avoid circular dependency
        from llm.tutor_model import _chat, _TUTOR_SYSTEM
        hyp = _chat(
            system="Write a short, factually correct paragraph answering the question. "
                   "Be concise (3-5 sentences). No headers.",
            user=query,
            max_tokens=200,
            temperature=0.3,
        )
        return hyp
    except Exception as e:
        print(f"[HyDE] Skipped: {e}")
        return None

def embed_query(text: str) -> np.ndarray:
    return embedding_model.encode([text], convert_to_numpy=True).astype("float32")


def search_index(query_embedding: np.ndarray, top_k: int = 8) -> tuple:
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]


def get_chunks(indices) -> list[str]:
    return [texts[i] for i in indices if i < len(texts)]


def keyword_search(query: str, top_k: int = 5) -> list[str]:
    if BM25_AVAILABLE and _bm25 is not None:
        tokens  = query.lower().split()
        scores  = _bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [texts[i] for i in top_idx if scores[i] > 0]
    #plain substing match
    return [t for t in texts if query.lower() in t.lower()][:top_k]

def _mmr(
    candidate_chunks: list[str],
    query_embedding: np.ndarray,
    top_k: int = 5,
    lambda_: float = 0.6,
) -> list[str]:

    if not candidate_chunks:
        return []

    embeddings = embedding_model.encode(candidate_chunks, convert_to_numpy=True)
    q          = query_embedding[0]

    norms       = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    emb_normed  = embeddings / norms
    q_normed    = q / (np.linalg.norm(q) + 1e-9)
    rel_scores  = emb_normed @ q_normed      

    selected_idx: list[int] = []
    remaining    = list(range(len(candidate_chunks)))

    for _ in range(min(top_k, len(candidate_chunks))):
        if not remaining:
            break
        if not selected_idx:
            best = max(remaining, key=lambda i: rel_scores[i])
        else:
            sel_emb = emb_normed[selected_idx]   
            scores  = []
            for i in remaining:
                red = np.max(emb_normed[i] @ sel_emb.T)  
                scores.append(lambda_ * rel_scores[i] - (1 - lambda_) * red)
            best = remaining[int(np.argmax(scores))]

        selected_idx.append(best)
        remaining.remove(best)

    return [candidate_chunks[i] for i in selected_idx]


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []
    pairs  = [[query, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def build_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)


def retrieve(query: str, top_k: int = 5, use_hyde: bool = True) -> str:
   
    all_chunks: list[str] = []


    for q in expand_query(query):
        emb     = embed_query(q)
        _, idxs = search_index(emb, top_k=top_k + 3)
        all_chunks.extend(get_chunks(idxs))

    if use_hyde:
        hyp_doc = _hyde_hypothetical_doc(query)
        if hyp_doc:
            hyp_emb = embed_query(hyp_doc)
            _, idxs = search_index(hyp_emb, top_k=top_k + 2)
            all_chunks.extend(get_chunks(idxs))


    all_chunks.extend(keyword_search(query, top_k=top_k))


    q_emb         = embed_query(query)
    diverse_chunks = _mmr(all_chunks, q_emb, top_k=top_k * 3)

    final_chunks = rerank(query, diverse_chunks, top_k=top_k)

    return build_context(final_chunks)