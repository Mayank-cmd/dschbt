import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import uuid

# --- Configuration ---
# (Preferably store these as secrets in Streamlit Cloud or a .env file)
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)

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
st.title("DocuBot - Ask Your Questions!!")

# State Initialization
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Navigation functions
def go_to_page(page_name):
    st.session_state.current_page = page_name
    st.experimental_rerun()

def start_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []
    go_to_page("chatbot")

# Sidebar for Chat History
st.sidebar.header("Chat History")
for chat_id, messages in st.session_state.chats.items():
    chat_label = f"Chat {list(st.session_state.chats.keys()).index(chat_id) + 1}"
    if st.sidebar.button(chat_label):
        st.session_state.current_chat = chat_id
        go_to_page("chatbot")
    
st.sidebar.button("New Chat", on_click=start_new_chat)

# Main Application Pages
if st.session_state.current_page == "home":
    if st.button("Go to Chatbot"):
        start_new_chat()
    if st.button("Ask ChatGPT"):
        go_to_page("chatgpt")

elif st.session_state.current_page == "chatbot":
    st.header("Chatbot - Ask Questions About Your PDF")

    current_chat = st.session_state.current_chat
    if current_chat not in st.session_state.chats:
        st.session_state.chats[current_chat] = []

    # Display previous messages
    for message in st.session_state.chats[current_chat]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # File Upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        raw_text = load_pdf(uploaded_file)
        add_to_vector_store(raw_text)
        st.success("PDF processed and added to the vector store!")

    # Chat Interface
    if prompt := st.chat_input("Your question"):
        user_message = {"role": "user", "content": prompt}
        st.session_state.chats[current_chat].append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
            answer = vector_index.query(prompt, llm=llm)
            assistant_message = {"role": "assistant", "content": answer}
            st.markdown(answer)
            st.session_state.chats[current_chat].append(assistant_message)

    if st.button("Back to Home"):
        go_to_page("home")

elif st.session_state.current_page == "chatgpt":
    st.header("ChatGPT - Ask Any Question")
    
    current_chat = st.session_state.current_chat
    if current_chat not in st.session_state.chats:
        st.session_state.chats[current_chat] = []

    # Display previous messages
    for message in st.session_state.chats[current_chat]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Interface
    if gpt_prompt := st.chat_input("Your question"):
        user_message = {"role": "user", "content": gpt_prompt}
        st.session_state.chats[current_chat].append(user_message)

        with st.chat_message("user"):
            st.markdown(gpt_prompt)

        with st.chat_message("assistant"):
            gpt_answer = llm(gpt_prompt)
            assistant_message = {"role": "assistant", "content": gpt_answer}
            st.markdown(gpt_answer)
            st.session_state.chats[current_chat].append(assistant_message)

    if st.button("Back to Home"):
        go_to_page("home")
