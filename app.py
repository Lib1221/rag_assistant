import streamlit as st
from rag_core import create_rag_pipeline

st.set_page_config(page_title="Intermediate RAG Assistant", page_icon="🤖")
st.title("Intermediate RAG Assistant with Memory")

qa, memory = create_rag_pipeline("documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about your documents:")

if query:
    with st.spinner("Thinking..."):
        # Generate answer
        result = qa.invoke({"query": query})["result"]
        
        # Save conversation
        st.session_state.chat_history.append({"question": query, "answer": result})
    
    # Display all conversation
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Assistant:** {chat['answer']}")
