from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import shutil
from pathlib import Path

# Imports from backend logic
from rag.new_rfp import ingest_new_rfps, delete_rfp_collection
from rag.session_doc import ingest_session_documents, delete_doc_collection
from src.normal_agent import stream_normal_agent
from src.tools import set_session_id

app = FastAPI(title="RFP Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    session_context: str

@app.post("/upload")
async def upload_files(
    session_id: str = Form(...),
    upload_type: str = Form(...),
    files: List[UploadFile] = File(...)
):
    # Save files to a temporary location
    temp_dir = Path("temp_uploads") / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    try:
        for file in files:
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))
            
        # Ingest the files based on the upload type
        if upload_type == "📄 New RFP":
            ingest_new_rfps(file_paths=file_paths, session_id=session_id)
        else:
            ingest_session_documents(file_paths=file_paths, session_id=session_id)
            
        # Set session id so tools can find the context
        set_session_id(session_id)
        
    finally:
        # Clean up the temporarily saved files after they have been ingested into ChromaDB
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

    return {"status": "success", "session_id": session_id}

@app.delete("/collection/{session_id}")
async def delete_collection(session_id: str, upload_type: str):
    try:
        if upload_type == "📄 New RFP":
            delete_rfp_collection(session_id=session_id)
        else:
            delete_doc_collection(session_id=session_id)
        return {"status": "deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(req: ChatRequest):
    def event_generator():
        try:
            for event in stream_normal_agent(
                req.message,
                session_id=req.session_id,
                session_context=req.session_context
            ):
                # Yield as JSON Lines (NDJSON) format
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps(("reasoning", f"Agent Error: {str(e)}")) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")
