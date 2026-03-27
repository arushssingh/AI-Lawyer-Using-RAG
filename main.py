import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
try:
    secrets = __import__("streamlit").secrets
    if "GROQ_API_KEY" in secrets:
        os.environ["GROQ_API_KEY"] = secrets["GROQ_API_KEY"]
    if "HF_API_KEY" in secrets:
        os.environ["HF_API_KEY"] = secrets["HF_API_KEY"]
except Exception:
    pass

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq



class HFEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.api_key = api_key
        self.api_url = f"https://router.huggingface.co/pipeline/feature-extraction/{model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=120
            )
            if response.status_code != 200:
                raise ValueError(f"HF API error {response.status_code}: {response.text}")
            all_embeddings.extend(response.json())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Question: {question} 
Context: {context} 
Answer:
"""


hf_api_key = os.environ.get("HF_API_KEY", "")
if not hf_api_key:
    st.error("HF_API_KEY is missing. Add it to .env locally or Streamlit Cloud Secrets.")
    st.stop()
embeddings = HFEmbeddings(api_key=hf_api_key)
FAISS_DB_PATH="vectorstore/db_faiss"


pdfs_directory = 'pdfs/'
os.makedirs(pdfs_directory, exist_ok=True)
os.makedirs('vectorstore/', exist_ok=True)
llm_model=ChatGroq(model="llama-3.3-70b-versatile")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def get_embedding_model():
    return HFEmbeddings(api_key=os.environ.get("HF_API_KEY", ""))


def create_vector_store(db_faiss_path, text_chunks):
    faiss_db=FAISS.from_documents(text_chunks, get_embedding_model())
    faiss_db.save_local(db_faiss_path)
    return faiss_db


def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)


def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})


uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)


user_query = st.text_area("Enter your prompt: ", height=150 , placeholder= "Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file and user_query:
        try:
            upload_pdf(uploaded_file)
            documents = load_pdf(pdfs_directory + uploaded_file.name)
            text_chunks = create_chunks(documents)
            faiss_db = create_vector_store(FAISS_DB_PATH, text_chunks)

            retrieved_docs=retrieve_docs(faiss_db, user_query)
            response=answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response.content)
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.error("Kindly upload a valid PDF file and/or ask a valid Question!")

