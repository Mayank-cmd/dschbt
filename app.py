import os
import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# --- Configuration ---
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)
pdf_directory = "uploaded_pdfs"

# --- File Management ---
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

pdf_metadata = {}  # Dictionary to store file paths and embedding status

# Get the document_id for the single PDF we're storing
if "uploaded_file_name" in st.session_state:
    document_id = st.session_state["uploaded_file_name"]
else:
    document_id = None

# Check if the file is uploaded, and store it if it hasn't been.
if document_id is None:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(pdf_directory, file_name)

        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state["uploaded_file_name"] = file_name
        document_id = file_name

pdf_metadata[document_id] = {"file_path": f"uploaded_pdfs/{document_id}", "embedded": False}


def load_and_embed_pdf(file_path, document_id):
    """Load, embed, and store a PDF if it hasn't been processed before."""
    if not pdf_metadata.get(document_id, {}).get("embedded", False):
        with open(file_path, "rb") as f:
            raw_text = load_pdf(f)
            add_to_vector_store(raw_text)
            pdf_metadata[document_id] = {"file_path": file_path, "embedded": True}


# --- Helper Functions ---
def load_pdf(uploaded_file):
    raw_text = ''
    pdfreader = PdfReader(uploaded_file)
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def add_to_vector_store(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vector_store.add_texts(texts)  # Add all texts

# --- Streamlit App ---
st.title("DataScience:GPT - PDF Q&A Chatbot")


# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Ask me questions about your uploaded PDF!"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if document_id in pdf_metadata:
        file_path = pdf_metadata[document_id]["file_path"]
        load_and_embed_pdf(file_path, document_id)  # Load and embed only if needed

        with st.chat_message("assistant"):
            vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
            answer = vector_index.query(prompt, llm=llm)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("No PDF found for this question. Please upload a PDF.")
