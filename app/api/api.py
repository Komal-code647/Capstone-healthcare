from fastapi import FastAPI
from pydantic import BaseModel

from app.crew.crew import run_healthcare_crew


app = FastAPI(
    title="Healthcare Knowledge Assistant",
    version="1.0"
)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():

    return {
        "message": "Healthcare Knowledge Assistant API Running"
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    response = run_healthcare_crew(request.query)

    return {
        "response": str(response)
    }
