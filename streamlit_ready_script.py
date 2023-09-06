
import streamlit as st

# Streamlit app title and introduction
st.title("Agile Marketing Q&A App")
st.write("Provide your question below and get answers from the book: Agile Marketing -from waterfall to water flow-.")

st.markdown("**Sample questions:**")
st.write("- What is Nero?")
st.write("- Why is it called Nero?")
st.write("- What are the advantages of Nero?")
st.write("- Who should read this?")
st.write("- What are the processes of Nero?")
st.write("- What is Nero Master?")
st.write("- Give me a list of reasons why Marketing should become Agile?")

# User Input for the question
query = st.text_input("Enter your question:", key="unique_query_key")

#Login to OpenAI using a file.
import os
import openai

# Fetch the API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key exists
if not api_key:
    raise ValueError("OpenAI API key not found!")

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

#import langchain libraries
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch

#load file
file = 'C:\\Users\\30698\Desktop\\test\\cleaned_updated_manual_sample.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')

#Convert into vector
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


#Generate response.
response = index.query(query)

st.write(response)


