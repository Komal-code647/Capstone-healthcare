from crewai import Task

from app.crew.agents import (
    symptom_analyzer_agent,
    medical_advisor_agent,
    emergency_agent
)


def create_tasks(user_query):

    symptom_analysis_task = Task(
        description=f"""
        Analyze the following patient symptoms carefully:

        {user_query}

        Identify possible conditions and important symptoms.
        """,
        expected_output="""
        A detailed symptom analysis with possible conditions.
        """,
        agent=symptom_analyzer_agent
    )

    medical_advice_task = Task(
        description=f"""
        Provide medical advice and precautions for:

        {user_query}

        Include lifestyle recommendations and precautions.
        """,
        expected_output="""
        Safe healthcare advice and recommendations.
        """,
        agent=medical_advisor_agent
    )

    emergency_task = Task(
        description=f"""
        Determine whether the following symptoms require emergency attention:

        {user_query}
        """,
        expected_output="""
        Emergency risk assessment and escalation advice.
        """,
        agent=emergency_agent
    )

    return [
        symptom_analysis_task,
        medical_advice_task,
        emergency_task
    ]
