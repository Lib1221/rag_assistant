# Architecture

A local retrieval-augmented generation assistant. Documents are indexed once at startup and queried through a Streamlit chat UI or a terminal loop.

## Flow

```
documents/*.txt|csv|pdf
   -> LangChain loaders (TextLoader, CSVLoader, PyPDFLoader)
   -> RecursiveCharacterTextSplitter (500 chars, 50 overlap)
   -> HuggingFaceEmbeddings (all-MiniLM-L6-v2)
   -> FAISS index (in memory)
   -> RetrievalQA chain ("stuff") with HuggingFacePipeline (flan-t5-small)
   -> Streamlit chat (app.py) or CLI (main.py)
```

## Modules

| File | Role |
| ---- | ---- |
| `rag_core.py` | `load_documents()` and `create_rag_pipeline()`; returns the QA chain and a `ConversationBufferMemory`. Skips unreadable files with a warning. |
| `app.py` | Streamlit page: custom CSS chat bubbles, suggestion buttons, session-state history, calls the pipeline per message. |
| `main.py` | Minimal terminal version that only loads `.txt` files. |

## Design notes

- Everything runs locally with open models; no API keys.
- The index is rebuilt on each start. For larger corpora, persist the FAISS index to disk (`vectorstore.save_local`) and load it if present.
- `flan-t5-small` keeps CPU latency low; swap the model name in `rag_core.py` for better answers on a GPU.
