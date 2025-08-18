# main.py
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

def load_documents(folder_path="documents"):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            docs.extend(loader.load())
    return docs

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

def main():
    qa = setup_rag("documents")
    print("RAG Assistant Ready! Type 'exit' to quit.\n")
    
    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        answer = qa.run(query)
        print("Answer:", answer, "\n")

if __name__ == "__main__":
    main()
