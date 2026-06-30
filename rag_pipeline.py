import os
import hashlib
import json
import time
from typing import List, Dict, Any, Tuple

# Retry decorator with exponential backoff for API calls.
# This prevents crashes on free-tier rate limits (15 RPM for Gemini free-tier).
def retry_with_backoff(max_retries: int = 6, initial_delay: float = 4.0, backoff_factor: float = 2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    err_msg = str(e)
                    # Retry on rate limits (429), temporary server issues (500, 503), or network timeouts
                    if any(code in err_msg for code in ["429", "ResourceExhausted", "Quota exceeded", "500", "503"]) or "timeout" in err_msg.lower():
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Auth errors (403, 400 invalid API key) or syntax/schema issues should fail immediately.
                        raise e
            raise last_exception
        return wrapper
    return decorator

# Custom, pure-Python vector store fallback in case ChromaDB installation fails
# on Python 3.13 due to missing C++ build tools.
class SimpleVectorStore:
    def __init__(self, persist_dir: str = "./simple_vector_store"):
        self.persist_dir = persist_dir
        self.metadata_path = os.path.join(persist_dir, "metadata.json")
        self.embeddings_path = os.path.join(persist_dir, "embeddings.npy")
        
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.embeddings: Any = None
        
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir, exist_ok=True)
        else:
            self._load()
            
    def _load(self):
        import numpy as np
        if os.path.exists(self.metadata_path) and os.path.exists(self.embeddings_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    self.metadatas = data.get("metadatas", [])
                    self.ids = data.get("ids", [])
                self.embeddings = np.load(self.embeddings_path)
            except Exception as e:
                # Fallback to empty if load fails due to corruption
                self.documents = []
                self.metadatas = []
                self.ids = []
                self.embeddings = None

    def _save(self):
        import numpy as np
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                    "ids": self.ids
                }, f, ensure_ascii=False, indent=2)
            if self.embeddings is not None:
                np.save(self.embeddings_path, self.embeddings)
        except Exception as e:
            pass

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        import numpy as np
        if not documents:
            return
        new_emb = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])
            
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        self._save()

    def delete(self, ids: List[str] = None, where: Dict[str, Any] = None):
        if not ids and not where:
            return
            
        indices_to_keep = []
        for idx, (doc_id, meta) in enumerate(zip(self.ids, self.metadatas)):
            keep = True
            if ids is not None and doc_id in ids:
                keep = False
            if where is not None:
                match = True
                for k, v in where.items():
                    val_to_check = v.get("$eq") if isinstance(v, dict) and "$eq" in v else v
                    if meta.get(k) != val_to_check:
                        match = False
                        break
                if match:
                    keep = False
            if keep:
                indices_to_keep.append(idx)
                
        if len(indices_to_keep) == len(self.ids):
            return
            
        self.documents = [self.documents[i] for i in indices_to_keep]
        self.metadatas = [self.metadatas[i] for i in indices_to_keep]
        self.ids = [self.ids[i] for i in indices_to_keep]
        
        if self.embeddings is not None and len(indices_to_keep) > 0:
            self.embeddings = self.embeddings[indices_to_keep]
        else:
            self.embeddings = None
            
        self._save()

    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        import numpy as np
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        dot_products = np.dot(self.embeddings, q_vec)
        norms_stored = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(q_vec)
        
        norms_stored[norms_stored == 0] = 1e-10
        if norm_query == 0:
            norm_query = 1e-10
            
        cosine_similarities = dot_products / (norms_stored * norm_query)
        cosine_distances = 1.0 - cosine_similarities
        
        # Sort indices by distance ascending
        top_k_indices = np.argsort(cosine_distances)[:k]
        
        return {
            "documents": [[self.documents[i] for i in top_k_indices]],
            "metadatas": [[self.metadatas[i] for i in top_k_indices]],
            "ids": [[self.ids[i] for i in top_k_indices]],
            "distances": [[float(cosine_distances[i]) for i in top_k_indices]]
        }

# Standard ChromaDB Wrapper
class ChromaVectorStore:
    def __init__(self, persist_dir: str = "./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="tax_clause_collection_local_v1",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def delete(self, ids: List[str] = None, where: Dict[str, Any] = None):
        self.collection.delete(ids=ids, where=where)

    def query(self, query_embedding: List[float], k: int = 4) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

def get_vector_store(persist_dir: str = "./vector_store", force_fallback: bool = False):
    if force_fallback:
        return SimpleVectorStore(persist_dir=persist_dir + "_simple")
    try:
        import chromadb
        return ChromaVectorStore(persist_dir=persist_dir + "_chroma")
    except Exception as e:
        print(f"ChromaDB not available. Using fallback: {e}")
        return SimpleVectorStore(persist_dir=persist_dir + "_simple")

# Sliding window character-level splitter.
# Word-boundary check ensures context doesn't split key financial terms or numbers in half.
def split_text_to_chunks(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
        
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # 1. Align the end of the chunk to a clean word boundary
        if end < len(text):
            boundary = -1
            # Search back up to 50 characters
            for i in range(end, max(start, end - 50), -1):
                if text[i - 1] in (' ', '\n', '\t'):
                    boundary = i
                    break
            if boundary != -1:
                end = boundary
                
        # 2. Align the start of the chunk to a clean word boundary (except for the first chunk)
        # This prevents truncated words (like 'quent' or 'lopment') at chunk boundaries
        actual_start = start
        if actual_start > 0:
            boundary_start = -1
            # Search backwards up to 30 characters for a word break
            for i in range(actual_start, max(0, actual_start - 30), -1):
                if text[i - 1] in (' ', '\n', '\t'):
                    boundary_start = i
                    break
            if boundary_start != -1:
                actual_start = boundary_start
            else:
                # Fallback: search forwards up to 30 characters for a word break
                for i in range(actual_start, min(len(text), actual_start + 30)):
                    if text[i] in (' ', '\n', '\t'):
                        actual_start = i + 1
                        break
                        
        chunk = text[actual_start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - overlap
        if start >= len(text) or end == len(text):
            break
        if start < 0:
            start = 0
        if start >= end:
            start = end
            
    return chunks

# Main RAG Pipeline orchestrator
class RAGPipeline:
    def __init__(self, api_key: str, persist_dir: str = "./vector_store_local"):
        from google import genai
        from sentence_transformers import SentenceTransformer
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key)
        self.db = get_vector_store(persist_dir=persist_dir)
        # Load local lightweight embedding model on CPU
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Meta file is co-located with the vector DB folder so they cannot drift out of sync
        self.meta_path = os.path.join(persist_dir + "_chroma", "vector_store_meta.json")
        self.indexed_files = self._load_meta()

    def _load_meta(self) -> Dict[str, Any]:
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_meta(self):
        os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
        try:
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(self.indexed_files, f, ensure_ascii=False, indent=2)
        except:
            pass

    def _generate_embedding(self, texts: List[str]) -> List[List[float]]:
        # Using local SentenceTransformer model (all-MiniLM-L6-v2) for CPU-based inference
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def index_pdf(self, file_name: str, file_bytes: bytes, file_hash: str) -> str:
        # Check if already indexed
        if file_hash in self.indexed_files:
            return "skipped"

        # Read PDF pages
        try:
            # Wrap bytes in a stream
            import io
            import pypdf
            pdf_stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(pdf_stream)
        except Exception as e:
            raise ValueError(f"Malformed or corrupt PDF file: {e}")

        # Check total pages
        num_pages = len(reader.pages)
        if num_pages == 0:
            raise ValueError("The uploaded PDF is empty (0 pages).")

        all_documents = []
        all_metadatas = []
        all_ids = []
        
        has_any_text = False
        for page_idx in range(num_pages):
            page = reader.pages[page_idx]
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                has_any_text = True
                # Split this page's text into chunks
                chunks = split_text_to_chunks(page_text)
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"{file_hash}_p{page_idx}_c{chunk_idx}"
                    all_documents.append(chunk)
                    all_metadatas.append({
                        "source": file_name,
                        "page": page_idx + 1,
                        "source_hash": file_hash
                    })
                    all_ids.append(chunk_id)

        if not has_any_text:
            raise ValueError("This PDF appears to contain only scanned images or has no extractable text.")

        # Batch embed: 14 chunks per request (down from 8) to cut API calls by ~43% while staying within free-tier payload limits
        batch_size = 14
        all_embeddings = []
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            embeddings = self._generate_embedding(batch_docs)
            all_embeddings.extend(embeddings)

        # Add to vector database
        self.db.add_documents(
            documents=all_documents,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )

        # Save to metadata index
        self.indexed_files[file_hash] = {
            "filename": file_name,
            "chunk_ids": all_ids
        }
        self._save_meta()
        return "indexed"

    def remove_pdf_by_hash(self, file_hash: str):
        if file_hash in self.indexed_files:
            # Delete from DB
            self.db.delete(where={"source_hash": file_hash})
            # Remove from tracking list
            del self.indexed_files[file_hash]
            self._save_meta()

    def sync_uploaded_files(self, current_uploads: List[Tuple[str, bytes]]) -> Dict[str, List[str]]:
        # Map current uploads to hashes
        current_hashes = {}
        for name, data in current_uploads:
            hasher = hashlib.sha256()
            hasher.update(data)
            current_hashes[hasher.hexdigest()] = (name, data)

        indexed_hashes = list(self.indexed_files.keys())
        
        added = []
        skipped = []
        removed = []

        # 1. Delete removed files
        for old_hash in indexed_hashes:
            if old_hash not in current_hashes:
                fname = self.indexed_files[old_hash]["filename"]
                self.remove_pdf_by_hash(old_hash)
                removed.append(fname)

        # 2. Add new files
        for file_hash, (name, data) in current_hashes.items():
            if file_hash in self.indexed_files:
                skipped.append(name)
            else:
                self.index_pdf(name, data, file_hash)
                added.append(name)

        return {
            "added": added,
            "skipped": skipped,
            "removed": removed
        }

    @retry_with_backoff()
    def _call_generation_model(self, system_instruction: str, prompt: str) -> str:
        from google.genai import types
        # Using gemini-2.5-flash as requested by the user.
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0  # Grounded RAG answer should be deterministic
            )
        )
        return response.text or ""

    def answer_query(self, query: str, k: int = 4) -> Dict[str, Any]:
        # 1. Embed query
        query_embeddings = self._generate_embedding([query])
        query_embedding = query_embeddings[0]

        # 2. Retrieve top-k context
        results = self.db.query(query_embedding, k=k)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Filter out results with no documents
        if not documents:
            return {
                "answer": "No documents found in the search index. Please upload documents first.",
                "sources": []
            }

        # 3. Build grounding context
        context_str = ""
        sources_list = []
        for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # Distances from HNSW cosine are 1 - similarity.
            # Convert to similarity score.
            sim_score = 1.0 - dist
            source_tag = f"[Source: {meta['source']}, Page: {meta['page']}]"
            context_str += f"--- CONTEXT CHUNK {idx+1} {source_tag} (Similarity: {sim_score:.4f}) ---\n{doc}\n\n"
            
            sources_list.append({
                "chunk_text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "similarity": sim_score
            })

        # 4. Formulate Prompt
        system_instruction = (
            "You are an expert financial and tax audit simplifying assistant. "
            "Your task is to simplify dense audit and tax clauses into plain, understandable English.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the query based ONLY on the provided context chunks. Do not assume or extrapolate anything.\n"
            "2. If the context does not contain enough information to answer the query, state: 'The provided document(s) do not contain sufficient information to answer this question.'\n"
            "3. Ground every single claim in the context chunks and cite its source. Place citations inline at the end of the relevant sentence or bullet point like: [Filename, Page X].\n"
            "4. NEVER use general knowledge or hallucinate. Be extremely strict.\n\n"
            "FORMATTING INSTRUCTIONS:\n"
            "5. Use Markdown for the response.\n"
            "6. Use bullet points only for itemized, categorical, or list-like information (e.g., types of investments, multiple line items). Do NOT force continuous reasoning into bullets.\n"
            "7. Use plain paragraph text for narrative or explanatory content — do not force paragraphs into bullets.\n"
            "8. Bold key figures, defined terms, or section headers within the answer for scannability.\n"
            "9. Do not mix formatting styles within a single logical point (e.g., don't half-bullet, half-paragraph the same idea).\n"
            "10. Maintain the existing citation format [filename, Page X] inline, but ensure citations attach cleanly to the relevant bullet or sentence rather than being awkwardly placed mid-list.\n"
            "11. Do not add a generic preamble or intro like 'Here is the answer:' — start directly with the substance."
        )

        prompt = f"Query: {query}\n\nRetrieved Context:\n{context_str}"

        # 5. Generate grounded response
        answer = self._call_generation_model(system_instruction, prompt)
        
        return {
            "answer": answer,
            "sources": sources_list
        }
