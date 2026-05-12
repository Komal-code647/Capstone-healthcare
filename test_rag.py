from app.rag.rag_pipeline import load_pipeline, ask_question

chain = load_pipeline()

response = ask_question(
    chain,
    "I have chest pain and shortness of breath"
)

print(response["answer"])
