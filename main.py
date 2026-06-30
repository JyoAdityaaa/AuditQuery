import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

app = FastAPI(title="AuditQuery API", version="1.0.0")

# Lazy-load RAG Pipeline to prevent crashes on startup if API key is missing
pipeline = None
init_error = None

def get_pipeline():
    global pipeline, init_error
    if pipeline is not None:
        return pipeline
        
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="GEMINI_API_KEY is not configured in the backend environment. Please check your .env file."
        )
        
    try:
        from rag_pipeline import RAGPipeline
        # Initialize RAG Pipeline targeting the local CPU embedding store
        pipeline = RAGPipeline(api_key=api_key, persist_dir="./vector_store_local")
        return pipeline
    except Exception as e:
        init_error = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG Pipeline: {init_error}"
        )

# API Request Models
class QueryRequest(BaseModel):
    query: str
    k: int = 4

# API Endpoints
@app.get("/api/documents")
def list_documents():
    """Retrieve list of indexed documents metadata."""
    try:
        pipe = get_pipeline()
        return {"documents": list(pipe.indexed_files.values())}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents and perform incremental sync."""
    try:
        pipe = get_pipeline()
        upload_tuples = []
        for file in files:
            content = await file.read()
            upload_tuples.append((file.filename, content))
            
        result = pipe.sync_uploaded_files(upload_tuples)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_documents(request: QueryRequest):
    """Answer question based on document context."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    try:
        pipe = get_pipeline()
        result = pipe.answer_query(request.query, k=request.k)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static File Routes (Serves the UI from the root)
@app.get("/")
def serve_landing():
    return FileResponse("landing.html")

@app.get("/landing.css")
def serve_landing_css():
    return FileResponse("landing.css")

@app.get("/app")
def serve_index():
    return FileResponse("index.html")

@app.get("/fonts.css")
def serve_fonts():
    return FileResponse("fonts.css")

@app.get("/styles.css")
def serve_css():
    return FileResponse("styles.css")

@app.get("/app.js")
def serve_js():
    return FileResponse("app.js")

if __name__ == "__main__":
    import uvicorn
    # Run server locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
