from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env loads correctly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not loaded!")

print(f"✅ API Key loaded: {api_key[:10]}...")

PDF_PATH = Path(__file__).parent / "Simon_Kete_Resume.pdf"
OUTPUT_DIR = "shared_faiss_index"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

text = extract_text_from_pdf(PDF_PATH)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_text(text)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local(OUTPUT_DIR)

print(f"✅ Saved vectorstore to {OUTPUT_DIR}")
