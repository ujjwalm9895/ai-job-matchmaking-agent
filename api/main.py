from fastapi import FastAPI
from pydantic import BaseModel
from api.model import JobMatchModel

app = FastAPI()
model = JobMatchModel()

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

