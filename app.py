import streamlit as st
import uuid
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import cassio

# --- Configuration ---
ASTRA_DB_TOKEN = 'your_astra_db_token'
ASTRA_DB_ID = 'your_astra_db_id'
OPENAI_API_KEY = 'your_openai_api_key'
TABLE_NAME = 'qa_mini_demo'

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)

# --- Helper Functions ---
def load_pdf(uploaded_file):
    raw_text = ''
    pdfreader = PdfReader(uploaded_file)
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def add_to_vector_store(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vector_store.add_texts(texts)  # Add all texts

# --- Streamlit App State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

if 'chats' not in st.session_state:
    st.session_state.chats = {}

# --- Navigation Function ---
def navigate_to(page):
    st.session_state.current_page = page

def start_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = []
    st.session_state.current_chat = chat_id
    navigate_to('chatbot')

# --- Sidebar for Chat History and New Chat ---
with st.sidebar:
    st.title("Chat History")
    for chat_id, messages in st.session_state.chats.items():
        if st.button(f"Chat {chat_id[:8]}"):
            st.session_state.current_chat = chat_id
            navigate_to('chatbot')
    if st.button("New Chat"):
        start_new_chat()

# --- Main App Views ---
if st.session_state.current_page == 'home':
    st.title("Home Page")
    st.write("Welcome to DocuBot!")
    if st.button("Go to Chatbot"):
        start_new_chat()
    if st.button("Ask ChatGPT"):
        navigate_to('chatgpt')

elif st.session_state.current_page == 'chatbot':
    st.title("Chatbot Page")
    st.write("Ask questions about your uploaded PDFs.")
    if "current_chat" in st.session_state and st.session_state.current_chat in st.session_state.chats:
        chat_id = st.session_state.current_chat
        messages = st.session_state.chats[chat_id]

        # Display previous messages
        for message in messages:
            role = message['role']
            content = message['content']
            st.markdown(f"**{role}:** {content}")

        # Input for new message
        prompt = st.text_input("Your question")
        if st.button("Send"):
            vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
            answer = vector_index.query(prompt, llm=llm)
            st.session_state.chats[chat_id].append({"role": "user", "content": prompt})
            st.session_state.chats[chat_id].append({"role": "assistant", "content": answer})
            st.experimental_rerun()  # Refresh to show new messages
    st.button("Back to Home", on_click=lambda: navigate_to('home'))

elif st.session_state.current_page == 'chatgpt':
    st.title("ChatGPT Page")
    st.write("Ask any question.")
    gpt_prompt = st.text_input("Your question for ChatGPT")
    if st.button("Send"):
        gpt_answer = llm(gpt_prompt)
        st.write(gpt_answer)
    st.button("Back to Home", on_click=lambda: navigate_to('home'))
