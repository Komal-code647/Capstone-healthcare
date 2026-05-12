from crewai import Agent, Task, Crew

from app.tools.rag_tool import MedicalRAGTool

rag_tool = MedicalRAGTool()

retriever_agent = Agent(
    role="Medical Retriever Agent",
    goal="Retrieve relevant medical evidence",
    backstory="Expert in medical retrieval systems",
    tools=[rag_tool],
    verbose=True
)

consultation_agent = Agent(
    role="Medical Consultation Agent",
    goal="Provide grounded medical consultation",
    backstory="Experienced clinical consultation specialist",
    tools=[rag_tool],
    verbose=True
)

diagnosis_agent = Agent(
    role="Diagnosis Support Agent",
    goal="Suggest possible medical conditions",
    backstory="Expert diagnostic assistant",
    tools=[rag_tool],
    verbose=True
)

validator_agent = Agent(
    role="Medical Validator Agent",
    goal="Validate medical correctness",
    backstory="Medical hallucination detection expert",
    tools=[rag_tool],
    verbose=True
)

def run_crew(query):

    retrieval_task = Task(
        description=f"""
        Retrieve relevant medical evidence
        for this query:

        {query}
        """,
        agent=retriever_agent
    )

    consultation_task = Task(
        description=f"""
        Generate medically grounded response
        using retrieved evidence.

        Query:
        {query}
        """,
        agent=consultation_agent
    )

    diagnosis_task = Task(
        description=f"""
        Suggest possible conditions based on:
        {query}
        """,
        agent=diagnosis_agent
    )

    validation_task = Task(
        description=f"""
        Validate final response.
        Remove hallucinations.
        Ensure answer comes only from retrieved context.
        """,
        agent=validator_agent
    )

    crew = Crew(
        agents=[
            retriever_agent,
            consultation_agent,
            diagnosis_agent,
            validator_agent
        ],

        tasks=[
            retrieval_task,
            consultation_task,
            diagnosis_task,
            validation_task
        ],

        verbose=True
    )

    result = crew.kickoff()

    return result
