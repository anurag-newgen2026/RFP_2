# RFP Intelligence Agent

An AI-powered application designed to ingest, align, and analyze Request for Proposals (RFPs). This tool leverages modern RAG (Retrieval-Augmented Generation) architectures to search historical RFPs and provide intelligent insights on new RFP documents.

## Architecture

This project features a decoupled client-server architecture:
- **Frontend**: A rich, interactive web UI built with Gradio (`app1.py`).
- **Backend**: A robust API server built with FastAPI (`main.py`) that handles ChromaDB document ingestion and streams agent reasoning logic via NDJSON.

## Quick Start

### 1. Setup Environment
Ensure you have Python installed, then create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
Open a terminal and run:
```bash
uvicorn main:app --reload --port 8000
```
*(The API documentation will be available at `http://localhost:8000/docs`)*

### 3. Start the Gradio Frontend
Open a **second** terminal, activate your virtual environment, and run:
```bash
python app1.py
```
*(The web UI will be available at `http://localhost:7860`)*

## Important Notes
- Ensure your `.env` file is properly configured with your required API keys (e.g., OpenAI, Tavily) before starting the servers.
- The `chroma_data/` directory is automatically generated upon your first document ingestion and is purposely excluded from source control due to file size.
