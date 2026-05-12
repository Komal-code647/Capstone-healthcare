from crewai import Crew

from app.crew.agents import (
    symptom_analyzer_agent,
    medical_advisor_agent,
    emergency_agent
)

from app.crew.tasks import create_tasks


def run_healthcare_crew(user_query):

    tasks = create_tasks(user_query)

    healthcare_crew = Crew(
        agents=[
            symptom_analyzer_agent,
            medical_advisor_agent,
            emergency_agent
        ],
        tasks=tasks,
        verbose=True
    )

    result = healthcare_crew.kickoff()

    return result
