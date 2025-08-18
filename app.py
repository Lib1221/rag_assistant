import os
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

@st.cache_resource
def load_documents(folder_path="documents"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

@st.cache_resource
def setup_rag(doc_folder="documents"):
    docs = load_documents(doc_folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(split_docs, embeddings)
    retriever = db.as_retriever()
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=generator)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("RAG-Powered Assistant (CPU-Friendly)")

qa = setup_rag("documents")
query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        answer = qa.invoke({"query": query})["result"]
    st.markdown(f"**Answer:** {answer}")
