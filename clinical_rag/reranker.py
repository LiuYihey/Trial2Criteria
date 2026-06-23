import torch
from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE cross-encoder reranker for PubMed semantic search."""

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        print(f"--- [INFO] Loading BGE reranker {model_name} on {self.device} ---")
        try:
            self.model = CrossEncoder(model_name, device=self.device)
        except OSError as exc:
            print(f"--- [WARN] Failed to load reranker on {self.device}: {exc} ---")
            print(f"--- [INFO] Retrying on cpu ---")
            self.device = "cpu"
            try:
                self.model = CrossEncoder(model_name, device=self.device)
            except Exception as exc2:
                print(f"--- [WARN] Failed to load reranker on cpu too: {exc2} ---")
                print(f"--- [INFO] Reranker unavailable, will skip reranking ---")
                self.model = None
        except Exception as exc:
            print(f"--- [WARN] Failed to load reranker: {exc} ---")
            print(f"--- [INFO] Reranker unavailable, will skip reranking ---")
            self.model = None

    def rerank(self, query: str, documents, top_k: int = 5):
        if not documents:
            return []
        if self.model is None:
            print("--- [INFO] Skipping reranking (model not loaded), returning top results directly ---")
            for i, doc in enumerate(documents[:top_k]):
                doc.metadata["bge_score"] = 1.0 - i * 0.01
            return documents[: min(top_k, len(documents))]
        doc_pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(doc_pairs)
        except Exception as exc:
            print(f"--- [ERROR] BGE reranking failed: {exc} ---")
            return documents[: min(top_k, len(documents))]
        for doc, score in zip(documents, scores):
            doc.metadata["bge_score"] = float(score)
        sorted_docs = sorted(
            documents, key=lambda item: item.metadata.get("bge_score", 0), reverse=True
        )
        return sorted_docs[: min(top_k, len(sorted_docs))]
