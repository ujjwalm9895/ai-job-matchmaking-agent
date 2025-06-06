from fastapi import FastAPI
from pydantic import BaseModel
from api.model import JobMatchModel
from api.graph import build_graph

app = FastAPI()
model = JobMatchModel()

graph = build_graph()

class JobDescription(BaseModel):
    text: str

@app.post("/match-resumes")
def match_resumes(job: JobDescription):
    matches = model.match_resumes(job.text)
    return {"matches": matches}

class ChatInput(BaseModel):
    prompt: str

@app.post("/chat")
def chat(input: ChatInput):
    response = model.chat(input.prompt)
    return {"response": response}

@app.post("/graph-match")
def run_graph(job: JobDescription):
    result = graph.invoke({"input": job.text})
    return {
        "matches": result.get("matches"),
        "llm_summary": result.get("summary")
    }