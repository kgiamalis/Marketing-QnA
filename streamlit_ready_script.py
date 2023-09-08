import streamlit as st
import os
import openai
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
    padding: 8px;
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

# Fetch the API key from Streamlit secrets
api_key = st.secrets["openai"]["openai_api_key"]

# Check if the API key exists
if not api_key:
    raise ValueError("OpenAI API key not found!")

#Add LLM
llm_model = "gpt-3.5-turbo"

# Load your file
file = 'cleaned_updated_manual_sample.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
).from_loaders([loader])

# User Input for the question
query = st.text_input("Ask your question:", key="unique_query_key",)

# Generate response
response = index.query(query)

# Replace the default response (if this is the behavior of index.query)
if response == "I don't know":
    response = "The info you asked is not part of my training dataset"

st.text("Here is the answer:")
# Display the "Here is the answer" text and response in the same block using markdown
st.markdown(f"**Here is the answer:**\n\n{response}", unsafe_allow_html=True)
st.write(response)

st.markdown("""
You can try one of the following questions:
- What is Nero?
- Why is it called Nero?
- What are the advantages of Nero?
- Who should read this?
- What are the processes of Nero?
- What is Nero Master?
- Give me a list of reasons why Marketing should become Agile?
""")
