# Job Matchmaking Agent

## Overview
AI-powered resume-job matching app with LLM, FAISS vector search, LangGraph orchestration, and Streamlit UI.

## Features
- Match resumes to job descriptions using vector embeddings + FAISS
- Chat with LLM fine-tuned using LoRA
- LangGraph pipeline for multi-step reasoning
- Streamlit frontend

## Usage
```bash
uvicorn api.main:app --reload
streamlit run frontend/app.py
