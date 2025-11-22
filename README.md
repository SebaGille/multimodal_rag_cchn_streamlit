# MULTIMODAL_RAG_CCHN_STREAMLIT

Lightweight multimodal RAG workflow for the CCHN Field Manual. The repo already ingests the short extract PDF, builds a FAISS vector store with OpenAI embeddings, exposes a CLI answer helper, and ships a Streamlit UI.

## Repository Layout

- `ingestion/parse_pdf.py` – uses Unstructured to convert PDFs into LangChain documents.
- `ingestion/build_vectorstore.py` – chunks parsed docs, embeds with OpenAI, and saves a local FAISS index.
- `rag/answer.py` – minimal retrieval + ChatOpenAI answering CLI with inline chunk citations.
- `app/streamlit_app.py` – single-question Streamlit interface showing the answer and retrieved snippets.
- `data/` – parsed document cache (e.g., `short_extract_docs.json`).
- `vectorstores/` – persisted FAISS indexes (e.g., `short_extract_faiss`).

## Prerequisites

1. Python 3.10+ and `pip`.
2. Install system tools once (macOS): `brew install poppler tesseract`.
3. Create a virtual environment and install requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
4. Create `.env` with `OPENAI_API_KEY=...`.

## Ingestion Workflow

Parse a PDF into JSON documents:
```bash
source .venv/bin/activate
python ingestion/parse_pdf.py \
  --pdf cchn_field_manual_short_extract.pdf \
  --output data/short_extract_docs.json
```

Build a FAISS store from those docs:
```bash
python ingestion/build_vectorstore.py \
  --docs data/short_extract_docs.json \
  --output vectorstores/short_extract_faiss
```

Adjust paths, chunk sizes, and models as needed via CLI flags.

## CLI Question Answering

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
python rag/answer.py \
  --question "What is a frontline negotiator?" \
  --vectorstore vectorstores/short_extract_faiss
```

The script prints the grounded answer and chunk references. The `KMP_DUPLICATE_LIB_OK` flag avoids macOS OpenMP duplication warnings when using FAISS + OpenAI embeddings.

## Streamlit App

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
streamlit run app/streamlit_app.py
```

Sidebar controls let you point to any FAISS directory, choose embedding/chat models, and select `k`. Provide an OpenAI API key in `.env` before launching.

## Scaling to the Full Manual

Reuse the same commands with the full `cchn_field_manual.pdf` to generate a second JSON + vector store, then point both the CLI and Streamlit UI to the new directory.

