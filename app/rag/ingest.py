from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")

def build_index():

    jq_schema = ".[] | {instruction: .instruction, input: .input, output: .output}"

    json_loader = JSONLoader(
        file_path=os.path.join(DATA_DIR, "chatdoctor5k.json"), 
        jq_schema=jq_schema,
        text_content=False
    )

    chatdoctor_docs = json_loader.load()

    csv_loader_1 = CSVLoader(
        file_path=os.path.join(DATA_DIR, "format_dataset.csv")
    )

    csv_loader_2 = CSVLoader(
        file_path=os.path.join(DATA_DIR, "small_articles.csv")
    )

    docs1 = csv_loader_1.load()
    docs2 = csv_loader_2.load()

    all_docs = chatdoctor_docs + docs1 + docs2

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Creating embeddings...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("Saving FAISS index...")

    vectorstore.save_local( os.path.join(BASE_DIR, "faiss_index"))

    print("FAISS index created successfully!")

if __name__ == "__main__":
    build_index()
