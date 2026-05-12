from app.crew.crew import run_healthcare_crew


query = """
I have severe chest pain, dizziness, sweating,
and shortness of breath.
"""

result = run_healthcare_crew(query)

print("\nFINAL RESPONSE:\n")
print(result)
