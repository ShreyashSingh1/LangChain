import os
import time
import streamlit as st  
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize session state for vector store
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2")
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
    
    try:
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    except Exception as e:
        st.error(f"Error loading documents: {e}")

# Set title
st.title("ChatGroq Demo")

# Initialize the language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Define prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Create LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# User input
prompt = st.text_input("Input your prompt here")

if prompt:
    with st.spinner("Processing..."):
        start = time.process_time()
        
        context = " ".join([doc.page_content for doc in st.session_state.final_documents])
        
        input_data = {
            "context": context,
            "input": prompt
        }
        
        try:
            response = llm_chain.invoke(input_data)  
            print("Response:", response)  
    
            if 'text' in response:
                st.write(f"Response time: {time.process_time() - start:.2f} seconds")
                st.write(response['text'])  
            else:
                st.error("Response does not contain a 'text' key.")
                
        except Exception as e:
            st.error(f"Error processing the request: {e}")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(st.session_state.final_documents):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content) 
                st.write("---------------------")

