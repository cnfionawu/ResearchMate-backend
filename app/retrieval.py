from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def hybrid_search(query, papers, top_k=5):
    # BM25 + FAISS hybrid search list of papers
    # Each paper structure: (id, title, authors, summary, source)
    
    abstracts = [p[3] for p in papers]
    # titles = [p[1] for p in papers]

    # BM25
    tokenized = [abstract.lower().split() for abstract in abstracts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    # SBERT + FAISS
    embeddings = sbert_model.encode(abstracts, convert_to_numpy=True)
    query_embedding = sbert_model.encode(query, convert_to_numpy=True).reshape(1, -1)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(query_embedding, top_k)

    # Normalize scores
    bm25_scores = np.array(bm25_scores)
    faiss_scores = np.zeros(len(bm25_scores))
    faiss_scores[I[0]] = 1 - (D[0] / np.max(D))  # convert distances to similarity

    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.ptp(bm25_scores) + 1e-6)
    hybrid_scores = 0.5 * bm25_norm + 0.5 * faiss_scores

    # Rank and return top results 
    # Pending: add thresholding and filtering
    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    return [papers[i] for i in ranked_indices]
