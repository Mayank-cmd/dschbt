import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# --- Configuration ---
ASTRA_DB_TOKEN = "your_astra_db_token"
ASTRA_DB_ID = "your_astra_db_id"
OPENAI_API_KEY = "your_openai_api_key"
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

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("DocuBot - Ask Your Questions!!"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(id='upload-data', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                           style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                                  'borderWidth': '1px', 'borderStyle': 'dashed',
                                  'borderRadius': '5px', 'textAlign': 'center'}, multiple=False), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='output-data-upload'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Input(id='input-question', type='text', placeholder='Your question', style={'width': '100%'}),
                width=10),
        dbc.Col(html.Button('Submit', id='submit-button', n_clicks=0), width=2)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='chat-output'), width=12)
    ])
])


@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filename, last_modified):
    if contents is not None:
        # Process the uploaded file and add to vector store
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        raw_text = load_pdf(decoded)
        add_to_vector_store(raw_text)
        return html.Div(['PDF processed and added to the vector store!'])


@app.callback(
    Output('chat-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input-question', 'value')
)
def update_chat(n_clicks, value):
    if n_clicks > 0 and value:
        vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)
        answer = vector_index.query(value, llm=llm)
        return html.Div([html.P(f"User: {value}"), html.P(f"DocuBot: {answer}")])

    return html.Div()


if __name__ == '__main__':
    app.run_server(debug=True)
