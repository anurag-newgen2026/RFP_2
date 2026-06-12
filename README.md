# RFP Intelligence Agent

An AI-powered application designed to ingest, align, and analyze Request for Proposals (RFPs). This tool leverages modern RAG (Retrieval-Augmented Generation) architectures to search historical RFPs and provide intelligent insights on new RFP documents.

## Architecture

This project features a unified architecture:
- **Application**: A rich, interactive web UI built with Gradio that handles both user interaction, ChromaDB ingestion, and agent reasoning logic internally (`app.py`).

## Quick Start

### 1. Setup Environment
Ensure you have Python installed, then create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Application
Open a terminal, activate your virtual environment, and run:
```bash
python app.py
```
*(The web UI will be available at `http://localhost:7860`)*

## Important Notes
- Ensure your `.env` file is properly configured with your required API keys (e.g., OpenAI, Tavily) before starting the application.
- The `chroma_data/` directory is automatically generated upon your first document ingestion and is purposely excluded from source control due to file size.
