"""NEXUS RAG — High Performance RAG Engine
Document loading, sentence-boundary chunking, SentenceTransformers embedding, and FAISS indexing.
"""

import time
import re
import hashlib
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# DOCUMENT LOADER
# ──────────────────────────────────────────────
class DocumentLoader:
    """Extracts raw text while strictly preserving page numbering and provenance."""

    @staticmethod
    def load(data: bytes, filename: str) -> Dict[str, Any]:
        pages_data: List[Dict[str, Any]] = []

        if filename.lower().endswith(".pdf"):
            try:
                with fitz.open(stream=data, filetype="pdf") as doc:
                    page_count = len(doc)
                    for i, page in enumerate(doc):
                        text = page.get_text() or ""
                        text = re.sub(r"\n{3,}", "\n\n", text).strip()
                        if text:
                            pages_data.append({
                                "page_num": i + 1,
                                "text": text,
                                "char_count": len(text)
                            })
            except Exception as e:
                return {"error": f"Failed to parse PDF '{filename}': {str(e)}"}
        else:
            # Plain text document
            try:
                content = data.decode("utf-8", errors="replace")
                raw_pages = content.split("---")
                if len(raw_pages) > 1:
                    for i, p in enumerate(raw_pages):
                        clean_text = p.strip()
                        if clean_text:
                            pages_data.append({
                                "page_num": i + 1,
                                "text": clean_text,
                                "char_count": len(clean_text)
                            })
                else:
                    chars_per_page = 2200
                    total_len = len(content)
                    page_idx = 1
                    for start in range(0, total_len, chars_per_page):
                        chunk_text = content[start:start + chars_per_page].strip()
                        if chunk_text:
                            pages_data.append({
                                "page_num": page_idx,
                                "text": chunk_text,
                                "char_count": len(chunk_text)
                            })
                            page_idx += 1
            except Exception as e:
                return {"error": f"Failed to parse text file '{filename}': {str(e)}"}

        total_pages = len(pages_data)
        if total_pages == 0:
            return {"error": f"Document '{filename}' appears to be empty."}

        full_text = "\n\n".join(p["text"] for p in pages_data)
        doc_hash = hashlib.sha256(data).hexdigest()[:16]

        return {
            "filename": filename,
            "hash": doc_hash,
            "pages": pages_data,
            "page_count": total_pages,
            "char_count": len(full_text),
            "size_bytes": len(data),
            "error": None
        }

# ──────────────────────────────────────────────
# TEXT CHUNKER
# ──────────────────────────────────────────────
class TextChunker:
    """Sentence-aware chunker that preserves page provenance."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        filename = doc_data["filename"]
        chunks: List[Dict[str, Any]] = []
        global_chunk_id = 1

        for page in doc_data["pages"]:
            page_num = page["page_num"]
            text = page["text"]

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            if len(paragraphs) <= 1:
                sentences = re.split(r"(?<=[.!?])\s+", text)
            else:
                sentences = []
                for p in paragraphs:
                    sentences.extend(re.split(r"(?<=[.!?])\s+", p))

            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append({
                            "chunk_id": global_chunk_id,
                            "source": filename,
                            "page": page_num,
                            "text": current_chunk,
                            "char_count": len(current_chunk)
                        })
                        global_chunk_id += 1

                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = f"{overlap_text} {sentence}".strip()
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append({
                    "chunk_id": global_chunk_id,
                    "source": filename,
                    "page": page_num,
                    "text": current_chunk,
                    "char_count": len(current_chunk)
                })
                global_chunk_id += 1

        return chunks

# ──────────────────────────────────────────────
# EMBEDDINGS & VECTOR STORE
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

class VectorStore:
    """FAISS-backed vector index.
    Embeddings are unit L2-normalized so Inner Product (IndexFlatIP) computes exact Cosine Similarity [-1.0, 1.0].
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        faiss.normalize_L2(query_embedding)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["score"] = float(score)  # Exact Cosine Similarity in [-1, 1]
                results.append(chunk_copy)
        return results

# ──────────────────────────────────────────────
# QUERY CLASSIFICATION & QUALITY HELPERS
# ──────────────────────────────────────────────
SUMMARY_TRIGGERS = [
    "what's in the document", "whats in the document", "what is in the document",
    "what is this document about", "what's this document about", "about the document",
    "what does the document say", "what does the document contain",
    "what is in this document", "what is the document about",
    "summarize", "summary", "overview", "key topics", "main concepts",
    "what are the contents", "table of contents", "explain the document",
    "give me an overview", "walk me through the document"
]

def is_summary_query(query: str) -> bool:
    """Detects broad document synthesis requests requiring multi-page representative coverage."""
    q = query.lower().strip()
    return any(trigger in q for trigger in SUMMARY_TRIGGERS)

def get_relevance_label(score: float) -> Tuple[str, str]:
    """Returns human-readable strength label and color for cosine similarity score."""
    if score >= 0.70:
        return "Strong", "#16A34A"
    elif score >= 0.50:
        return "Good", "#2563EB"
    elif score >= 0.30:
        return "Moderate", "#D97706"
    else:
        return "Weak", "#94A3B8"

# ──────────────────────────────────────────────
# UNIFIED RAG PIPELINE
# ──────────────────────────────────────────────
class RAGPipeline:
    """End-to-end pipeline encapsulating document ingestion, indexing, and retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedder = get_embedding_model(model_name)
        self.vector_store = VectorStore(dimension=384)
        self.documents: Dict[str, Dict[str, Any]] = {}

    @property
    def is_indexed(self) -> bool:
        """True only if at least one document is indexed with active vector chunks."""
        return len(self.documents) > 0 and len(self.vector_store.chunks) > 0

    def ingest_document(
        self,
        data: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 80
    ) -> Dict[str, Any]:
        """Loads, chunks, embeds, and indexes a document."""
        start_time = time.perf_counter()

        # Step 1: Load
        doc_data = DocumentLoader.load(data, filename)
        if doc_data.get("error"):
            return {"success": False, "error": doc_data["error"]}

        # Step 2: Chunk
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(doc_data)
        if not chunks:
            return {"success": False, "error": "No valid text chunks could be generated."}

        # Step 3: Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        # Step 4: Index
        self.vector_store.add_embeddings(embeddings, chunks)

        elapsed = time.perf_counter() - start_time
        doc_data["chunk_count"] = len(chunks)
        doc_data["indexing_time"] = round(elapsed, 3)

        self.documents[filename] = doc_data

        return {
            "success": True,
            "filename": filename,
            "page_count": doc_data["page_count"],
            "chunk_count": len(chunks),
            "indexing_time": round(elapsed, 3),
            "error": None
        }

    def retrieve(self, query: str, top_k: int = 4, threshold: float = 0.22) -> Dict[str, Any]:
        """Encodes query and retrieves top chunks with grounding confidence check and hybrid lexical re-ranking."""
        start_time = time.perf_counter()

        if not self.is_indexed:
            return {
                "query": query,
                "chunks": [],
                "top_score": 0.0,
                "is_grounded": False,
                "is_summary": False,
                "threshold": threshold,
                "retrieval_latency_ms": 0.0,
                "total_chunks_available": 0
            }

        is_summary = is_summary_query(query)

        if is_summary:
            # Broad synthesis request: retrieve a diverse set of representative chunks across pages
            all_chunks = self.vector_store.chunks
            if len(all_chunks) <= 8:
                retrieved_chunks = [dict(c) for c in all_chunks]
                for c in retrieved_chunks:
                    c["score"] = 0.85
            else:
                # Sample representative chunks across unique pages
                pages_seen = set()
                sampled_chunks = []
                for c in all_chunks:
                    p = c.get("page", 1)
                    if p not in pages_seen:
                        chunk_copy = dict(c)
                        chunk_copy["score"] = 0.82
                        sampled_chunks.append(chunk_copy)
                        pages_seen.add(p)
                    if len(sampled_chunks) >= 6:
                        break
                retrieved_chunks = sampled_chunks or [dict(c) for c in all_chunks[:6]]

            elapsed = time.perf_counter() - start_time
            return {
                "query": query,
                "chunks": retrieved_chunks,
                "top_score": 0.85,
                "is_grounded": True,
                "is_summary": True,
                "threshold": threshold,
                "retrieval_latency_ms": round(elapsed * 1000, 2),
                "total_chunks_available": len(all_chunks)
            }

        # Specific question: Dense vector retrieval + Hybrid Lexical Boost
        query_emb = self.embedder.encode([query], show_progress_bar=False, convert_to_numpy=True)
        retrieved_chunks = self.vector_store.search(query_emb, top_k=min(top_k * 3, 12))

        # Hybrid lexical re-ranking to boost exact entity/keyword matches
        stopwords = {
            "what", "is", "the", "a", "an", "and", "or", "in", "of", "to", "for",
            "with", "on", "at", "by", "from", "who", "which", "how", "why", "when",
            "where", "tell", "me", "about", "can", "you", "does", "do", "explain"
        }
        q_words = [w.lower().strip("?,.!:;\"'") for w in query.split() if len(w) > 2]
        keywords = [w for w in q_words if w not in stopwords]

        for chunk in retrieved_chunks:
            chunk_text_lower = chunk.get("text", "").lower()
            if keywords:
                matches = sum(1 for kw in keywords if kw in chunk_text_lower)
                lexical_overlap = matches / len(keywords)
                # Boost dense score with lexical match (up to +0.25)
                chunk["score"] = min(1.0, chunk["score"] + (lexical_overlap * 0.25))

        # Re-sort after hybrid score boost and truncate to top_k
        retrieved_chunks.sort(key=lambda c: c["score"], reverse=True)
        retrieved_chunks = retrieved_chunks[:top_k]

        elapsed = time.perf_counter() - start_time

        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0
        # Strict grounding: only consider evidence sufficient if top_score >= threshold
        is_grounded = top_score >= threshold and len(retrieved_chunks) > 0

        # Filter out chunks that fall well below threshold
        filtered_chunks = [c for c in retrieved_chunks if c["score"] >= max(0.12, threshold * 0.65)]

        return {
            "query": query,
            "chunks": filtered_chunks if is_grounded else retrieved_chunks,
            "top_score": round(top_score, 4),
            "is_grounded": is_grounded,
            "is_summary": False,
            "threshold": threshold,
            "retrieval_latency_ms": round(elapsed * 1000, 2),
            "total_chunks_available": len(self.vector_store.chunks)
        }

    def clear(self) -> None:
        """Resets the vector index and document registry."""
        self.vector_store = VectorStore(dimension=384)
        self.documents.clear()
