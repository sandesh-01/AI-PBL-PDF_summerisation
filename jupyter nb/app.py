import streamlit as st
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfFileReader
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GooglePalm
from htmlTemplates import css, bot_template, user_template

def get_pdfs_text(pdfs_files):
    text=""
    for pdf in pdfs_files:
        pdf_reader=PdfFileReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n\n\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,   
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embedding= GooglePalmEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding)
    return vectorstore

def get_converation(vectorstore):
    llm=GooglePalm(google_api_key=os. environ[ "GOOGLE_API_KEY"], temperature=0)

    memory=ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
        page_title="AI PBL PROJECT",
        page_icon=":middle_finger:",
    )
    st.write(css, unsafe_allow_html=True)

    if "convesation" not in st.session_state:
        st.session_state.convesation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("CHAT WITH MULTIPLE PDF FILES")
    user_question = st.text_input("Question")
    if user_question:
        handle_userinput(user_question)
        
    st.write(user_template.replace("{{MSG}}", "Hello Bost"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("YOUR DOCUMENTS ")
        pdfs_files=st.file_uploader("Upload your pdf files", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner("Uploading"):
                #input 
                raw_text=get_pdfs_text(pdfs_files)
                #chunks
                text_chunks = get_text_chunks(raw_text)
                #vectors 
                vectorstore = get_vectorstore(text_chunks)
                #conversation memory
                st.session_state.convesation = get_converation(vectorstore)

             
            

if __name__ == '__main__':
    main()
    