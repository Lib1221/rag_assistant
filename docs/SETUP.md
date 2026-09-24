# Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install sentence-transformers transformers torch pypdf
```

Add your files to `documents/` (`.txt`, `.csv`, `.pdf`), then:

```bash
streamlit run app.py     # web chat
python main.py           # terminal
```

First run downloads the embedding and generator models from the HuggingFace Hub.

## Troubleshooting

- Slow answers: reduce chunk count with a smaller corpus or a lighter generator.
- PDF errors: make sure `pypdf` is installed; scanned PDFs need OCR first.
