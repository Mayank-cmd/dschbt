import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
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

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    raw_text = load_pdf(uploaded_file)
    add_to_vector_store(raw_text)
    st.success("PDF processed and added to the vector store!")

# Accuracy Testing
if st.button("Run Accuracy Test"):
    if "test_data.csv" in st.secrets:
        test_data = pd.read_csv(st.secrets["test_data.csv"])
        vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
        
        true_answers = test_data["answer"].tolist()
        predictions = []
        
        for question in test_data["question"]:
            prediction = vector_index.query(question, llm=llm)
            predictions.append(prediction)
        
        correct_predictions = sum(1 for true, pred in zip(true_answers, predictions) if true == pred)
        total_predictions = len(predictions)
        accuracy = correct_predictions / total_predictions
        
        st.write(f"Accuracy: {accuracy}")
        
        # Plot accuracy
        fig, ax = plt.subplots()
        ax.bar(["Accuracy"], [accuracy], color='skyblue')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Chatbot Accuracy')
        st.pyplot(fig)
    else:
        st.error("No test data file found. Please upload 'test_data.csv' to secrets.")

# Displaying the app messages
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
