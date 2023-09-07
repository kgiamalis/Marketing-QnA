import streamlit as st
import os
import openai
import datetime
import time
import random
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator

# Embed the custom CSS directly in the Streamlit app
custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    background-color: #f7f8fa;
}
.stMarkdown {
    background-color: #ffffff;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
}
.stButton > button {
    background-color: #0079bf;
    border-radius: 8px;
    color: white;
    border: none;
}
"""
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

# Streamlit app title and introduction
st.title("Agile Marketing Q&A")
st.write("Ask your question and get answers based on the book: Agile Marketing -from waterfall to water flow-.")
st.write("What is Nero?",'\n'
             "Why is it called Nero?","\n"
             "What are the advantages of Nero?",'\n'
             "Who should read this?",'\n'
             "What are the processes of Nero?",'\n'
             "What is Nero Master?",'\n'
             "Give me a list of reasons why Marketing should become Agile?")

# User Input for the question
query = st.text_input("Enter your question:", key="unique_query_key")

# Fetch the API key from Streamlit secrets
api_key = st.secrets["openai"]["openai_api_key"]

# Check if the API key exists
if not api_key:
    raise ValueError("OpenAI API key not found!")

# Account for deprecation of LLM model by setting the model based on the current date
current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
llm_model = "gpt-3.5-turbo" if current_date > target_date else "gpt-3.5-turbo-0301"

# Load your file
file = 'cleaned_updated_manual_sample.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')

# Fetch the API key from Streamlit's secrets
api_key = st.secrets["openai"]["openai_api_key"]

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
).from_loaders([loader])

# Generate response
response = index.query(query)

st.write(response)
