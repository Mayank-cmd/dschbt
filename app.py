import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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

# --- Authentication ---
def login(username, password):
    # Example hardcoded credentials
    users = {
        "user1": "password1",
        "user2": "password2"
    }
    return users.get(username) == password

# --- Streamlit App ---
st.title("DocuBot - Ask Your Questions!!")

# Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")
else:
    # Navigation buttons
    if "page" not in st.session_state:
        st.session_state.page = "home"

    def go_to_home():
        st.session_state.page = "home"
        st.experimental_set_query_params(page="home")

    def go_to_chatbot():
        st.session_state.page = "chatbot"
        st.experimental_set_query_params(page="chatbot")

    def go_to_chatgpt():
        st.session_state.page = "chatgpt"
        st.experimental_set_query_params(page="chatgpt")

    # Check if there are query params and update page state
    query_params = st.experimental_get_query_params()
    if "page" in query_params:
        st.session_state.page = query_params["page"][0]

    if st.session_state.page == "home":
        if st.button("Go to Chatbot"):
            go_to_chatbot()
        if st.button("Ask ChatGPT"):
            go_to_chatgpt()

    elif st.session_state.page == "chatbot":
        st.header("Chatbot - Ask Questions About Your PDF")
        # File Upload
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            raw_text = load_pdf(uploaded_file)
            add_to_vector_store(raw_text)
            st.success("PDF processed and added to the vector store!")

        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Ask me questions about your uploaded PDF!"})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
                answer = vector_index.query(prompt, llm=llm)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        if st.button("Back to Home"):
            go_to_home()

    elif st.session_state.page == "chatgpt":
        st.header("ChatGPT - Ask Any Question")
        # Chat Interface
        if "gpt_messages" not in st.session_state:
            st.session_state.gpt_messages = []
            st.session_state.gpt_messages.append({"role": "assistant", "content": "Ask me anything!"})

        for message in st.session_state.gpt_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if gpt_prompt := st.chat_input("Your question"):
            st.session_state.gpt_messages.append({"role": "user", "content": gpt_prompt})

            with st.chat_message("user"):
                st.markdown(gpt_prompt)

            with st.chat_message("assistant"):
                gpt_answer = llm(gpt_prompt)
                st.markdown(gpt_answer)
                st.session_state.gpt_messages.append({"role": "assistant", "content": gpt_answer})

        if st.button("Back to Home"):
            go_to_home()
