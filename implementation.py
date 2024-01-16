import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import  json

load_dotenv()
DB_FAISS_PATH = 'vectorstore/MagicCompRules'

#1. Load Vector Database
def retrieve_info(query):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    similar_response = db.similarity_search(query, k=5)

    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# 2. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k-0613")
template = """
You are a Magic: the Gathering judge whose purpose is to answer user queries related to Magic: the Gathering.

How to behave:
- Only respond to user messages which are related to the game of Magic: the Gathering
- Answer any queries users may have related to Magic: the Gathering
- Answer any queries regarding rulings and in-game interactions between Magic: the Gathering cards
- You should always reference official game rules whenever possible, which are provided below
- If unsure, tell the user that you cannot provide an accurate answer

Here's a list of relevant offical game rules:
{rulings}

Please answer the user's question to the best of your ability

User Question:
{message}
"""

prompt = PromptTemplate(
    input_variables=["message", "rulings"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# retrieve_info("What happens when a 2/1 with first strike is blocked by 2 1/1s?")

def generate_response(message):
    rulings = retrieve_info(message)
    print("Rulings: ", rulings)
    res = chain.run(message=message, rulings=rulings)
    print("Response: ", res)
    return res

generate_response("What happens when a 2/1 with first strike is blocked by 2 1/1s?")
