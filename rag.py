import streamlit as st 
import os
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain_huggingface import HuggingFaceEmbeddings #to get embedding model
from langchain.schema import Document #to store the text and metadata
from langchain.text_splitter import CharacterTextSplitter #to split the raw text into chunks
from langchain_community.vectorstores import FAISS #to store the embedded data for the similarity search

key = os.getenv('gemini_api_key')
genai.configure(api_key=key)

gemini_model = genai.GenerativeModel('gemini-2.0-flash')

st.cache_data(show_spinner='Loading Embedding Model... ')

def load_embedding():
    return HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')

with st.spinner('Loading Embedding model..... '):
    embedding_model = load_embedding()

st.header('RAG Assistant :green[Using Embedding & Gemini LLM]')
st.subheader('Your Interlligent Document Assistant')

st.write('Done')

uploaded_file = st.file_uploader('Upload the document here in PDF format', type=['pdf'])

if uploaded_file:
    st.write('Uploaded Successfully')
    
if uploaded_file:
    pdf=PdfReader(uploaded_file)
    raw_text=''
    
    for page in pdf.pages:
        raw_text += page.extract_text()
        
    st.write('Extracted successfully')
    
    if raw_text.strip():#strip will remove spaces - front and back
        doc = Document(page_content = raw_text)
        splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        chunk_text = splitter.split_documents([doc])
        
        text = [i.page_content for i in chunk_text]
    
        vector_db = FAISS.from_texts(text,embedding_model)
        retrieve = vector_db.as_retriever()
    
        st.success('Document processed successfully....Ask question now.')
        
        query = st.text_input('Enter your query here:')
        
        if query:
            with st.chat_message('human'):
                
                with st.spinner('Analyzing the document'):
                    relevant_docs = retrieve.get_relevant_documents(query)
                    
                    content = '\n\n'.join([i.page_content for i in relevant_docs])
                    
                    prompt = f'''
                    You are an AI expert, Use the content given to answer the query asked
                    by the user. If you are unsure you should say 'Iam unsure about the
                    question asked'
                    content : {content}
                    
                    query : {query}
                    
                    result : '''
                    
                    response = gemini_model.generate_content(prompt)
                    
                    st.markdown('### :green[Result]')
                    st.write(response.text)
                    
    else:
        
        st.warning('Drop the file in the proper pdf format')
            

        
        