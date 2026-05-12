from fastapi import FastAPI
from pydantic import BaseModel

from app.agents.crew import run_crew

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health():
    return {"message": "Healthcare RAG API Running"}

@app.post("/query")
def query_rag(request: QueryRequest):

    response = run_crew(request.query)

    return {
        "query": request.query,
        "response": str(response)
    }
