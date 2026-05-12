import time
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from app.rag.prompts import SYSTEM_PROMPT
import os
load_dotenv()

def load_pipeline():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
      model_name="openai/gpt-oss-20b:free",
      temperature=0.3,
      max_tokens=2000,
      openai_api_base="https://openrouter.ai/api/v1",
      openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=f"""
        {SYSTEM_PROMPT}

        Context:
        {{context}}

        Chat History:
        {{chat_history}}

        Question:
        {{question}}

        Answer:
        """
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

def ask_question(chain, question):

    start = time.time()

    result = chain.invoke({
        "question": question
    })

    latency = time.time() - start

    docs = result["source_documents"]

    retrieved_docs = [
        doc.page_content[:300]
        for doc in docs
    ]

    return {
        "answer": result["answer"],
        "retrieved_docs": retrieved_docs,
        "latency": latency
    }
