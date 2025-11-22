# MULTIMODAL_RAG_CCHN_STREAMLIT

Lightweight multimodal RAG workflow for the full CCHN Field Manual. The repo ingests the complete PDF, builds a FAISS vector store with OpenAI embeddings, exposes a CLI answer helper, and ships a Streamlit UI.

## Repository Layout

- `ingestion/parse_pdf.py` – uses Unstructured to convert PDFs into LangChain documents.
- `ingestion/build_vectorstore.py` – chunks parsed docs, embeds with OpenAI, and saves a local FAISS index.
- `rag/answer.py` – minimal retrieval + ChatOpenAI answering CLI with inline chunk citations.
- `app/streamlit_app.py` – single-question Streamlit interface showing the answer and retrieved snippets.
- `data/` – raw PDFs and parsed document cache (e.g., `raw/CCHN-Field-Manual-EN.pdf`, `full_manual_docs.json`).
- `vectorstores/` – persisted FAISS indexes (e.g., `full_manual_faiss`).

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

Parse the full manual into JSON documents (≈2.5k elements):
```bash
source .venv/bin/activate
python ingestion/parse_pdf.py \
  --pdf data/raw/CCHN-Field-Manual-EN.pdf \
  --output data/full_manual_docs.json
```

Build a FAISS store from those docs:
```bash
python ingestion/build_vectorstore.py \
  --docs data/full_manual_docs.json \
  --output vectorstores/full_manual_faiss
```

Adjust paths, chunk sizes, and models as needed via CLI flags.

> Need the lightweight extract for demos? Swap the `--pdf`, `--output`, and `--docs` arguments to point back to `cchn_field_manual_short_extract.pdf` and its derived assets.

## CLI Question Answering

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
python rag/answer.py \
  --question "What is a frontline negotiator?" \
  --vectorstore vectorstores/full_manual_faiss
```

The script prints the grounded answer and chunk references. The `KMP_DUPLICATE_LIB_OK` flag avoids macOS OpenMP duplication warnings when using FAISS + OpenAI embeddings.

## Streamlit App

```bash
KMP_DUPLICATE_LIB_OK=TRUE \
streamlit run app/streamlit_app.py
```

Sidebar controls let you point to any FAISS directory, choose embedding/chat models, and select `k`. Provide an OpenAI API key in `.env` before launching.  
The repository already tracks `vectorstores/full_manual_faiss`, so remote hosts (e.g., Streamlit Cloud) clone the ready-to-use index by default.

## Scaling to the Full Manual

The default configuration already ships with `data/full_manual_docs.json` and expects `vectorstores/full_manual_faiss`. Re-run the ingestion commands whenever the source PDF changes. If you maintain multiple indexes (e.g., extract vs. full manual), update the Streamlit **Engine → Vectorstore directory** field or pass a different `--vectorstore` path to the CLI.

