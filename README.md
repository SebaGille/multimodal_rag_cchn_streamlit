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
5. (macOS) Unify the OpenMP runtime once per install so FAISS and Torch use the same `libomp.dylib`:
   ```bash
   source .venv/bin/activate
   python scripts/unify_openmp.py
   ```

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
python rag/answer.py \
  --question "What is a frontline negotiator?" \
  --vectorstore vectorstores/full_manual_faiss
```

The script prints the grounded answer and chunk references.

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Sidebar controls let you point to any FAISS directory, choose embedding/chat models, and select `k`. Provide an OpenAI API key in `.env` before launching.  
The repository already tracks `vectorstores/full_manual_faiss`, so remote hosts (e.g., Streamlit Cloud) clone the ready-to-use index by default.

## OpenMP Runtime Health Check (macOS)

FAISS (`faiss-cpu==1.13.0`) and Torch (`torch==2.9.1`) each bundle `libomp.dylib`. Loading both copies crashes the process with `OMP: Error #15`. Keep the environment healthy by:

1. Activating `.venv` and running `python scripts/unify_openmp.py` whenever dependencies are installed or upgraded.
2. Verifying the fix via the regression harness:
   ```bash
   source .venv/bin/activate
   python3 tests/mode_option_regression.py
   ```
   The suite finishes without setting `KMP_DUPLICATE_LIB_OK`, and the same setup applies to the CLI and Streamlit flows.

## Scaling to the Full Manual

The default configuration already ships with `data/full_manual_docs.json` and expects `vectorstores/full_manual_faiss`. Re-run the ingestion commands whenever the source PDF changes. If you maintain multiple indexes (e.g., extract vs. full manual), update the Streamlit **Engine → Vectorstore directory** field or pass a different `--vectorstore` path to the CLI.

