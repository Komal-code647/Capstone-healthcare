from crewai import Agent

from langchain_openai import ChatOpenAI

from app.tools.rag_tool import healthcare_rag_tool
import os

symptom_analyzer_agent = Agent(
    llm="openrouter/openai/gpt-oss-20b:free",
    role="Symptom Analyzer",
    goal="""
    Analyze patient symptoms and identify possible medical conditions.
    Use the RAG tool to retrieve relevant medical knowledge.
    """,
    backstory="""
    You are an experienced healthcare AI assistant specialized in
    understanding symptoms and retrieving medical insights.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[healthcare_rag_tool]
)


medical_advisor_agent = Agent(
    llm="openrouter/openai/gpt-oss-20b:free",
    role="Medical Advisor",
    goal="""
    Provide safe healthcare guidance and recommendations based on
    retrieved medical information.
    """,
    backstory="""
    You are a healthcare advisory AI focused on patient safety,
    medical guidance, precautions, and next steps.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[healthcare_rag_tool]
)


emergency_agent = Agent(
    llm="openrouter/openai/gpt-oss-20b:free",
    role="Emergency Response Agent",
    goal="""
    Detect medical emergencies and recommend urgent medical attention
    when symptoms are severe.
    """,
    backstory="""
    You are trained to identify dangerous symptoms and escalate
    critical medical situations immediately.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[healthcare_rag_tool]
)
