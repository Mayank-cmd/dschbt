

# --- Configuration ---
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# --- Evaluation Setup ---
from langchain.evaluation.qa import QAEvalChain




# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)
qa_eval_chain = QAEvalChain.from_llm(llm, chain_type="stuff")  # Initialize evaluation chain

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

# --- Accuracy Evaluation Function ---
def evaluate_accuracy(pdf_text, questions, expected_answers):
    correct = 0
    total = len(questions)

    vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)  # Use uploaded PDF

    for question, expected_answer in zip(questions, expected_answers):
        prediction = vector_index.query(question, llm=llm)
        graded_output = qa_eval_chain.evaluate(
            input_text=pdf_text,
            prediction=prediction,
            answer=expected_answer
        )
        if graded_output["text"] == "CORRECT":
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy


# --- Streamlit App ---
st.title("DataScience:GPT - PDF Q&A Chatbot")

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


# --- Evaluation Section ---
if st.button("Evaluate Accuracy"):
    # 1. Load Your Questions and Expected Answers
    questions_file = st.file_uploader("test.csv", type=["csv", "txt"])
    if questions_file is not None:
        import pandas as pd
        questions_df = pd.read_csv(questions_file)
        questions = questions_df["question"].tolist()
        expected_answers = questions_df["answer"].tolist()

        # 2. Evaluate and Display
        accuracy = evaluate_accuracy(raw_text, questions, expected_answers)
        st.write(f"Accuracy: {accuracy:.2f}%")

