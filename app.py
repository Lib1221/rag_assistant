import streamlit as st
import time
from rag_core import create_rag_pipeline

# ---------- Page Config ----------
st.set_page_config(
    page_title="⚡ Neon RAG Chat",
    page_icon="🤖",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; font-family: 'Segoe UI', sans-serif; }

#chat-container { width: 700px; max-width: 90%; margin: auto; display: flex; flex-direction: column; }

.user-bubble { background: linear-gradient(135deg, #00ffcc, #00ffff); color: black; padding: 10px 15px; 
    border-radius: 15px; margin: 5px 0; max-width: 80%; float: right; clear: both; font-weight: bold; box-shadow: 0 0 15px #00ffff; }
.assistant-bubble { background: linear-gradient(135deg, #ff00cc, #ff66ff); color: white; padding: 10px 15px; 
    border-radius: 15px; margin: 5px 0; max-width: 80%; float: left; clear: both; font-weight: bold; box-shadow: 0 0 15px #ff00cc; }

#chat-messages { max-height: 500px; overflow-y: auto; padding: 10px; }

input, button { font-size: 16px; }

.suggestion-btn { display: flex; flex-direction: column; margin-top: 20px; }
.suggestion-btn button { background: linear-gradient(135deg, #00ffff, #00ffcc); color: black; border: none; 
    padding: 8px 12px; margin: 4px 0; border-radius: 12px; font-weight: bold; box-shadow: 0 0 10px #00ffff; transition: 0.3s; }
.suggestion-btn button:hover { background: linear-gradient(135deg, #ff00cc, #ff66ff); color: white; box-shadow: 0 0 12px #ff00cc; }

</style>

<script>
function scrollChat() {
    const chat = document.querySelector("#chat-messages");
    if(chat) { chat.scrollTop = chat.scrollHeight; }
}
</script>
""", unsafe_allow_html=True)

# ---------- Load RAG ----------
qa, memory = create_rag_pipeline("documents")

# ---------- Session State ----------
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = [{"question": "", "answer": "Hello! I am your RAG assistant 🤖. Ask me anything about your documents."}]
if "current_query" not in st.session_state: st.session_state.current_query = ""
if "first_render" not in st.session_state: st.session_state.first_render = True

# ---------- Suggested Questions ----------
sample_questions = [
    "Summarize the document",
    "Key points of the document?",
    "Who are the main people mentioned?",
    "Explain simply",
    "Document conclusion?"
]

# ---------- Sidebar with Suggested Questions ----------
st.sidebar.markdown("### Suggested Questions")
for idx, q in enumerate(sample_questions):
    if st.sidebar.button(q, key=f"suggest_{idx}"):
        st.session_state.current_query = q
        st.experimental_rerun()

# ---------- Render Chat ----------
chat_box = st.empty()
title_box = st.empty()
input_box = st.empty()

def render_chat():
    with chat_box.container():
        st.markdown('<div id="chat-container">', unsafe_allow_html=True)
        st.markdown('<div id="chat-messages">', unsafe_allow_html=True)

        for chat in st.session_state.chat_history:
            if chat["question"]:
                st.markdown(f'<div class="user-bubble">🙋 {chat["question"]}</div>', unsafe_allow_html=True)
            if chat["answer"]:
                st.markdown(f'<div class="assistant-bubble">🤖 {chat["answer"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Scroll to bottom
        st.markdown('<script>scrollChat()</script>', unsafe_allow_html=True)

# ---------- Send Message ----------
def send_message(user_input):
    st.session_state.chat_history.append({"question": user_input, "answer": ""})
    render_chat()

    with st.spinner("🤖 Thinking..."):
        answer = qa.invoke({"query": user_input})["result"]

    # Typing animation
    answer_placeholder = st.empty()
    typed_text = ""
    for char in answer:
        typed_text += char
        answer_placeholder.markdown(f'<div class="assistant-bubble">🤖 {typed_text}</div>', unsafe_allow_html=True)
        st.markdown('<script>scrollChat()</script>', unsafe_allow_html=True)
        time.sleep(0.02)

    st.session_state.chat_history[-1]["answer"] = answer
    render_chat()

# ---------- Layout ----------
if st.session_state.first_render:
    # First time: show big title + centered input
    title_box.markdown("## ⚡ Neon RAG Chat")
    with input_box.form(key="chat_form_first", clear_on_submit=True):
        user_input = st.text_input("Type your question here:", placeholder="Ask me anything...")
        submit_button = st.form_submit_button("Send")
        if submit_button and user_input.strip():
            st.session_state.first_render = False
            send_message(user_input)
else:
    # After first question: title top, input bottom
    title_box.markdown("## ⚡ Neon RAG Chat")
    with input_box.form(key="chat_form_bottom", clear_on_submit=True):
        user_input = st.text_input("", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Send")
        if submit_button and user_input.strip():
            send_message(user_input)

render_chat()
