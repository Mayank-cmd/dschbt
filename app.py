import streamlit as st
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import os

ASTRA_DB_TOKEN = st.secrets["astra_db_token"]
ASTRA_DB_ID = st.secrets["astra_db_id"]
OPENAI_API_KEY = st.secrets["openai_api_key"]
TABLE_NAME = "qa_mini_demo"

cassio.init(token=ASTRA_DB_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

vector_store = Cassandra(embedding=embedding, table_name=TABLE_NAME)

def load_pdf(uploaded_file):
    raw_text = ''
    pdfreader = PdfReader(uploaded_file)
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def add_to_vector_store(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)
    vector_store.add_texts(texts)

def preprocess_text(text):
    return text.strip().lower()

st.title("DataScience:GPT - PDF Q&A Chatbot")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    raw_text = load_pdf(uploaded_file)
    add_to_vector_store(raw_text)
    st.success("PDF processed and added to the vector store!")

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

if st.button("Run Accuracy Test and Visualize Keywords"):
    if os.path.exists("test_data.csv"):
        test_data = pd.read_csv("test_data.csv")
        st.write(test_data.head())

        required_columns = ["question", "answer"]
        for column in required_columns:
            if column not in test_data.columns:
                st.error(f"Missing required column: {column}")
                st.stop()

        if "accuracy_history" not in st.session_state:
            st.session_state.accuracy_history = []

        predictions = []
        true_answers = test_data["answer"].tolist()
        vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

        for question in test_data["question"]:
            prediction = vector_index.query(question, llm=llm)
            predictions.append(prediction)

        true_embeddings = model.encode(true_answers, convert_to_tensor=True)
        pred_embeddings = model.encode(predictions, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(true_embeddings, pred_embeddings)

        accuracy_threshold = 0.7
        correct_predictions = (similarities.diag() > accuracy_threshold).sum().item()
        accuracy = correct_predictions / len(true_answers)
        st.write(f"Accuracy: {accuracy}")

        st.session_state.accuracy_history.append(accuracy)

        # Plot accuracy
        fig1, ax1 = plt.subplots()
        ax1.plot(st.session_state.accuracy_history, marker='o', linestyle='-', color='skyblue')
        ax1.set_xlabel('Test Instance')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Over Time')
        ax1.set_ylim(0, 1) 
        st.pyplot(fig1)

        # Keyword visualization
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(test_data["question"])  
        feature_names = vectorizer.get_feature_names_out()

        top_n_keywords = 10 
        top_keywords_indices = tfidf_matrix.toarray().argsort(axis=1)[:, -top_n_keywords:]

        x = []
        y = []
        keywords = []
        for i, question in enumerate(test_data["question"]):
            for j in top_keywords_indices[i]:
                x.append(i + 1)  
                y.append(tfidf_matrix[i, j])  
                keywords.append(feature_names[j])

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter = ax2.scatter(x, y, c='skyblue', alpha=0.7)

        ax2.set_xlabel('Question Number')
        ax2.set_ylabel('TF-IDF Score')
        ax2.set_title('Top Keywords in Questions')
        ax2.set_xticks(range(1, len(test_data["question"]) + 1))
        
        st.pyplot(fig2)

    else:
        st.error("The file 'test_data.csv' was not found. Please upload the file and try again.")
