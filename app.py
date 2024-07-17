import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Configuration ---
ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

# --- Initialization ---
cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)
lemmatizer = WordNetLemmatizer()

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

def preprocess_text(text):
    text = text.strip().lower()
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

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

# ... (Your other imports and helper functions remain the same)

# --- Accuracy Testing ---
if st.button("Run Accuracy Test"):
    if os.path.exists("test_data.csv"):
        test_data = pd.read_csv("test_data.csv")
        st.write(test_data.head())

        # Verify column names
        required_columns = ["question", "answer"]
        for column in required_columns:
            if column not in test_data.columns:
                st.error(f"Missing required column: {column}")
                st.stop()

        predictions = []
        true_answers = test_data["answer"].tolist()
        vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

        for question in test_data["question"]:
            prediction = vector_index.query(question, llm=llm)
            predictions.append(prediction)

        # Calculate semantic similarity
        true_embeddings = model.encode(true_answers, convert_to_tensor=True)
        pred_embeddings = model.encode(predictions, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(true_embeddings, pred_embeddings)

        threshold = 0.7
        correct_predictions = (similarities.diag() > threshold).sum().item()
        accuracy = correct_predictions / len(true_answers)
        st.write(f"Accuracy (Semantic Similarity): {accuracy:.2f}")

        # Preprocess answers for exact match
        true_answers_preprocessed = [preprocess_text(answer) for answer in true_answers]
        predictions_preprocessed = [preprocess_text(prediction) for prediction in predictions]

        # Convert preprocessed answers to numerical labels (1 for match, 0 for mismatch)
        true_labels = [1 if true == pred else 0 for true, pred in zip(true_answers_preprocessed, predictions_preprocessed)]

        # Calculate Precision and Recall
        precision = precision_score(true_labels, true_labels, average='micro') 
        recall = recall_score(true_labels, true_labels, average='micro')

        st.write(f"Precision (Exact Match): {precision:.2f}")
        st.write(f"Recall (Exact Match): {recall:.2f}")

        # Debugging Output
        for i in range(len(true_answers)):
            st.write(f"Q: {test_data['question'][i]}")
            st.write(f"True: {true_answers[i]}")
            st.write(f"Pred: {predictions[i]}")
            st.write("---")

        # Plotting the metrics
        metrics = ['Accuracy', 'Precision', 'Recall']
        scores = [accuracy, precision, recall]

        fig, ax = plt.subplots()
        ax.barh(metrics, scores, color=['blue', 'orange', 'green'])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score')
        ax.set_title('Chatbot Performance Metrics')

        st.pyplot(fig)
    else:
        st.error("The file 'test_data.csv' was not found. Please upload the file and try again.")

