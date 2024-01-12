import streamlit as st
from dotenv import load_dotenv
import openpyxl as xl

from langchain.document_loaders.csv_loader import CSVLoader, DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

def vectorize_csv(filename):
    loader = CSVLoader(file_path=filename)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)    

# Get input file from user

def main():
    st.set_page_config(
        page_title="Vectorize Your DAta", page_icon=":bird:")

    st.header("Vector Embeddings")

    uploaded_file = st.file_uploader("Upload dataset")
    if uploaded_file:
        st.write("filename:", uploaded_file.name)
    
    if st.button("Vectorize Data"):
        if uploaded_file.name.endswith('.csv'):
            # langchain loader requires file to be saved locally
            wb = xl.load_workbook(uploaded_file)
            wb.save()
            vectorize_csv(uploaded_file.name)
    


if __name__ == '__main__':
    main()