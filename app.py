import streamlit as st
from dotenv import load_dotenv
import os.path

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/'

def vectorize_csv(filename):
    st.write("Vectorizing CSV...")
    loader = CSVLoader(file_path=DATA_PATH+filename)
    documents = loader.load()

    st.write(documents)
    # embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(documents, embeddings)
    # FAISS_NAME = filename.split(".")[0]
    # db.save_local(DB_FAISS_PATH+FAISS_NAME)

def vectorize_pdf(filename):
    st.write("Vectorizing PDF...")
    loader = PyPDFLoader(DATA_PATH+filename)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    st.write("Generating Embeddings")
    embeddings = OpenAIEmbeddings()
    st.write("Creating FAISS DB")
    db = FAISS.from_documents(texts, embeddings)
    FAISS_NAME = filename.split(".")[0]
    db.save_local(DB_FAISS_PATH+FAISS_NAME)
    st.write("Vector Store Complete")
def save_uploadedfile(uploadedfile):
    with open(os.path.join("./data/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

# Get input file from user
def main():
    st.set_page_config(
        page_title="Vectorize Your DAta", page_icon=":bird:")

    st.header("Vector Embeddings")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "pdf"])
    if uploaded_file:
        st.write("filename:", uploaded_file.name)
    
    if st.button("Vectorize Data"):
        # langchain loader requires file to be saved locally
        save_uploadedfile(uploaded_file)
        if uploaded_file.name.endswith('.csv'):
            vectorize_csv(uploaded_file.name)
        if uploaded_file.name.endswith('.pdf'):
            vectorize_pdf(uploaded_file.name)
    


if __name__ == '__main__':
    main()