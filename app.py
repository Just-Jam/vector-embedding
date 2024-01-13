import streamlit as st
from dotenv import load_dotenv
import os.path

from langchain.document_loaders.csv_loader import CSVLoader, DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

def vectorize_csv(filename):
    loader = CSVLoader(file_path=filename)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("./datasets/", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

# Get input file from user
def main():
    st.set_page_config(
        page_title="Vectorize Your DAta", page_icon=":bird:")

    st.header("Vector Embeddings")

    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "pdf", "docx"])
    if uploaded_file:
        st.write("filename:", uploaded_file.name)
    
    if st.button("Vectorize Data"):
        if uploaded_file.name.endswith('.csv'):
            # langchain loader requires file to be saved locally
            save_uploadedfile(uploaded_file)
            vectorize_csv(uploaded_file.name)
    


if __name__ == '__main__':
    main()