from crewai.tools import BaseTool

from app.rag.rag_pipeline import load_pipeline, ask_question


chain = load_pipeline()


class HealthcareRAGTool(BaseTool):
    name: str = "Healthcare RAG Tool"
    description: str = (
        "Answers healthcare-related questions using the RAG pipeline."
    )

    def _run(self, query: str) -> str:

        response = ask_question(chain, query)

        return response["answer"]


healthcare_rag_tool = HealthcareRAGTool()
