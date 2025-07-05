import streamlit as st
from streamlit.components.v1 import html
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import os

def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_dotenv()

load_css("style.css")

st.set_page_config(page_title="AltairGPT")
if "messages" not in st.session_state:
    st.session_state.messages = []

if "waiting_for_answer" not in st.session_state:
    st.session_state.waiting_for_answer = False

if "added_pdf_names" not in st.session_state:
    st.session_state.added_pdf_names = set()

api_key = os.getenv("OPENAI_API_KEY")

#PDF_PATH = "Simon_Kete_Resume.pdf"
PDF_PATH = os.path.join(os.path.dirname(__file__), "Simon_Kete_Resume.pdf")

FAISS_INDEX_PATH = "faiss_index"

import fitz
import os

def extract_text_from_pdf(pdf_source):
    """
    Accepts either:
    - a string file path (local file)
    - or a Streamlit UploadedFile object
    Returns extracted text.
    """
    if isinstance(pdf_source, str):
        # It's a path string
        if not os.path.exists(pdf_source):
            raise FileNotFoundError(f"File not found: {pdf_source}")
        doc = fitz.open(pdf_source)
    else:
        # Assume it's a file-like object (uploaded file)
        pdf_bytes = pdf_source.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text()
    return text


SHARED_FAISS_INDEX_PATH = "shared_faiss_index"

@st.cache_resource(show_spinner=True)
def load_vectorstore(api_key: str):
    if os.path.exists(SHARED_FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(
            SHARED_FAISS_INDEX_PATH,
            OpenAIEmbeddings(openai_api_key=api_key),
            allow_dangerous_deserialization=True
        )
    else:
        raise FileNotFoundError("Shared FAISS index not found. Run build_vectorstore.py first.")
    return vectorstore

def add_pdf_to_vectorstore(pdf_file, vectorstore):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name
    
    pdf_text = extract_text_from_pdf(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(pdf_text)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    vectorstore.add_texts(chunks, embeddings=embeddings)
    
    # vectorstore.save_local(FAISS_INDEX_PATH)  # optional, if you want to persist changes

    return vectorstore


def add_message(role, content):
    st.session_state.messages.append((role, content))

def get_answer(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content

def render_messages():
    st.empty()

    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='user-msg'>😊 You: {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='gpt-msg'>🤖 GPT: {msg}</div>", unsafe_allow_html=True)

def render_pdf_font():
    if "added_pdf_names" in st.session_state and st.session_state.added_pdf_names:
        st.markdown("Uploaded PDFs:")
        for pdf_name in st.session_state.added_pdf_names:
            st.write(f"- {pdf_name}")
    else:
        st.write("No PDFs uploaded yet.")

title = st.container()
with title:
    st.write("""<div class='fixed-title'></div>""", unsafe_allow_html=True)
    st.title("AltairGPT")

chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0)

header = st.container()

with header:
    
    st.write("""<div class='fixed-header'></div>""", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore(api_key)

    if uploaded_file:
        filename = uploaded_file.name
        if filename not in st.session_state.added_pdf_names:
            st.session_state.vectorstore = add_pdf_to_vectorstore(uploaded_file, st.session_state.vectorstore)
            st.session_state.added_pdf_names.add(filename)
            
    #render_pdf_font()

    retriever = st.session_state.vectorstore.as_retriever()

    user_input = st.text_input(
        "Your message", 
        key="chat_input", 
        label_visibility="collapsed", 
        placeholder=st.session_state.get("placeholder", "Type your message..."),
        value=st.session_state.get("chat_input", "")  # set current input value
    )
    
    if st.button("Send"):
        add_message("user", user_input)
        st.session_state.waiting_for_answer = True

render_messages()

# If waiting for GPT answer, generate it and add it, then reset flag and rerun
if st.session_state.waiting_for_answer:
    with st.spinner("Thinking..."):
        answer = get_answer(st.session_state.messages[-1][1])
    
    add_message("assistant", answer)
    role, msg = st.session_state.messages[-1]
    st.markdown(f"<div class='gpt-msg'>🤖 GPT: {msg}</div>", unsafe_allow_html=True)
    st.session_state.waiting_for_answer = False