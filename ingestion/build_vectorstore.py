"""
Vector store builder for the CCHN manual RAG prototype.

Usage:
    python ingestion/build_vectorstore.py \
        --docs data/full_manual_docs.json \
        --output vectorstores/full_manual_faiss
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Allow running this script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.parse_pdf import load_documents


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Create overlapping text chunks to balance retrieval recall and precision."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(list(documents))


def build_faiss_store(
    documents: List[Document],
    output_dir: Path,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Embed documents and persist a FAISS index."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(documents, embedding=embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=str(output_dir))


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build a FAISS vector store from parsed docs.")
    parser.add_argument(
        "--docs",
        type=Path,
        required=True,
        help="Path to JSON documents produced by parse_pdf.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("vectorstores/full_manual_faiss"),
        help="Directory where the FAISS index will be stored.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Character length for text chunks.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Number of overlapping characters between chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )

    args = parser.parse_args()

    docs = load_documents(args.docs)
    chunked_docs = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    build_faiss_store(
        documents=chunked_docs,
        output_dir=args.output,
        embedding_model=args.embedding_model,
    )
    print(f"Stored {len(chunked_docs)} chunks in {args.output}")


if __name__ == "__main__":
    main()

