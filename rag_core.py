import os
import warnings
from langchain.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Load documents with error handling
def load_documents(folder="documents"):
    docs = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            if filename.endswith(".txt"):
                docs.extend(TextLoader(path).load())
            elif filename.endswith(".csv"):
                docs.extend(CSVLoader(path).load())
            elif filename.endswith(".pdf"):
                docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            warnings.warn(f"Skipping {filename} due to error: {e}")
    return docs

# Create RAG pipeline
def create_rag_pipeline(folder="documents"):
    docs = load_documents(folder)
    
    # Split long documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    retriever = vectorstore.as_retriever()
    
    # LLM generator
    generator = pipeline("text2text-generation", model="google/flan-t5-small")
    llm = HuggingFacePipeline(pipeline=generator)
    
    # Conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # RAG QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    
    return qa, memory
