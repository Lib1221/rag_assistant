# RAG Assistant

A local retrieval-augmented generation (RAG) assistant that answers questions about your own documents. Drop `.txt`, `.csv`, or `.pdf` files into the `documents/` folder and chat with them through a Streamlit UI or a terminal prompt. Everything runs locally with open models, so no API key is required.

## How it works

1. Documents in `documents/` are loaded with LangChain loaders (`TextLoader`, `CSVLoader`, `PyPDFLoader`).
2. Text is split into 500-character chunks with 50-character overlap.
3. Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS vector index.
4. A `RetrievalQA` chain retrieves the most relevant chunks and generates an answer with `google/flan-t5-small`.
5. `ConversationBufferMemory` keeps chat history across turns in the web UI.

## Project structure

```
rag_assistant/
├── app.py            # Streamlit chat interface
├── main.py           # Minimal terminal version (txt files only)
├── rag_core.py       # Document loading, indexing, and pipeline creation
├── documents/        # Put your source files here
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/Lib1221/rag_assistant.git
cd rag_assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers transformers torch pypdf
```

Run the web UI:

```bash
streamlit run app.py
```

Or the terminal version:

```bash
python main.py
```

## Tech stack

Python, LangChain, FAISS, HuggingFace Transformers, Sentence Transformers, Streamlit.

## Notes

The default generator (`flan-t5-small`) is chosen so the project runs on a laptop CPU. Swap the model name in `rag_core.py` for a larger model if you have a GPU or want better answers.

## Documentation

Developer docs live in [`docs/`](docs/):

- [Architecture](docs/ARCHITECTURE.md)
- [Setup](docs/SETUP.md)
- [Contributing](docs/CONTRIBUTING.md)
