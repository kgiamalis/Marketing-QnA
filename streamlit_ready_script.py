import streamlit as st
import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
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

# Streamlit app title and introduction
st.title("Agile Marketing Q&A")

st.markdown(""" Ask your question and get answers based on my book: Agile Marketing -from waterfall to water flow-. 
You can try one of the following questions:
- What is Nero?
- Why is it called Nero?
- What are the advantages of Nero?
- Who should read this?
- What are the processes of Nero?
- What is Nero Master?
- Give me a list of reasons why Marketing should become Agile?
""")

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
query = st.text_input("Ask your question:",key="unique_query_key")

# Generate response
response = index.query(query)

# Replace the default response (if this is the behavior of index.query)
if response == "I don't know":
    response = "The info you asked is not part of my training dataset."

# Display the "Here is the answer" text and response in the same block using markdown
st.markdown(f"**Here is the answer:**\n\n{response}", unsafe_allow_html=True)

# Add a horizontal line as a divider
st.markdown('---')

# Add some space
st.write('\n')

# Add "Buy My Book" and "Feedback Form" links side by side
buy_book_link = '[Buy the book](https://www.amazon.de/-/en/Konstantinos-Giamalis/dp/6180023735/?&_encoding=UTF8&tag=kgiamalis-21&linkCode=ur2&linkId=ba4eaff10ab7d658db964e48125abc7d&camp=1638&creative=6742)'
feedback_form_link = '[Feedback form](https://forms.gle/S8zK7dRR6sKAoUe97)'

# Display the links
st.markdown(f"{buy_book_link} &nbsp;&nbsp;&nbsp;&nbsp; {feedback_form_link}", unsafe_allow_html=True)
