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

# --- Configuration ---
# (Store these as secrets or environment variables)
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)
qa_eval_chain = QAEvalChain.from_llm(llm, chain_type="stuff")

# --- Helper Functions ---
def load_pdf(uploaded_file):
    raw_text = ""
    pdfreader = PdfReader(uploaded_file)
    for page in pdfreader.pages:
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

    vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

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
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    if st.session_state["uploaded_file"] is None or st.session_state["uploaded_file"].name != uploaded_file.name:
        st.session_state["uploaded_file"] = uploaded_file
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
if "questions_file" not in st.session_state:
    st.session_state["questions_file"] = None
    
if st.button("Evaluate Accuracy"):
    # 1. Get Questions File 
    if st.session_state["questions_file"] is None:
        questions_file = st.file_uploader("Upload questions file (CSV/text)", type=["csv", "txt"])
        if questions_file is not None:
            st.session_state["questions_file"] = questions_file

    if st.session_state["questions_file"] is not None:        
        try:
            questions_file = st.session_state["questions_file"]
            import pandas as pd

            # Detect file type and read accordingly
            if questions_file.name.endswith(".csv"):
                questions_df = pd.read_csv(questions_file)
            elif questions_file.name.endswith(".txt"):
                lines = questions_file.readlines()
                questions = [line.decode('utf-8').split(",")[0].strip() for line in lines]
                expected_answers = [line.decode('utf-8').split(",")[1].strip() for line in lines]
                questions_df = pd.DataFrame({"question": questions, "answer": expected_answers})
            else:
                raise ValueError("Unsupported file format. Please use CSV or TXT.")

            # Check if required columns are present
            if "question" not in questions_df.columns or "answer" not in questions_df.columns:
                raise ValueError("Questions file must have 'question' and 'answer' columns.")

            questions = questions_df["question"].tolist()
            expected_answers = questions_df["answer"].tolist()

            # 2. Evaluate and Display
            with st.spinner("Evaluating..."):  # Add a spinner to show progress
                accuracy = evaluate_accuracy(raw_text, questions, expected_answers)
            st.success(f"Accuracy: {accuracy:.2f}%")  # Display with success message

        except FileNotFoundError:
            st.error(f"File not found: {questions_file.name}")
        except ValueError as e:
            st.error(f"Error reading questions file: {e}")
