"""
PDF parsing utilities for the CCHN manual RAG pipeline.

Usage:
    python ingestion/parse_pdf.py \
        --pdf data/raw/cchn_field_manual_short_extract.pdf \
        --output data/short_extract_docs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf


def _ensure_json_safe(value: Any) -> Any:
    """Recursively convert metadata to JSON-serializable primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _ensure_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_json_safe(v) for v in value]
    return str(value)


def element_to_document(element: Any) -> Optional[Document]:
    """Convert an Unstructured element into a LangChain Document."""
    text = getattr(element, "text", "") or ""
    raw_metadata = getattr(element, "metadata", None)
    if hasattr(raw_metadata, "to_dict"):
        metadata = raw_metadata.to_dict()
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}

    base_metadata: Dict[str, Any] = {
        "element_type": getattr(element, "category", None),
        "page_number": metadata.get("page_number"),
        "source": metadata.get("filename"),
        "coordinates": metadata.get("coordinates"),
    }

    # Capture image references, if present.
    image_path = metadata.get("image_path")
    if image_path:
        base_metadata["image_path"] = image_path

    # Drop empty text blocks.
    if not text.strip() and not base_metadata.get("image_path"):
        return None

    return Document(page_content=text.strip(), metadata=base_metadata)


def parse_pdf(
    pdf_path: Path,
    chunk_size: Optional[int] = None,
    max_characters: Optional[int] = None,
    include_images: bool = True,
) -> List[Document]:
    """
    Parse a PDF into LangChain Documents using Unstructured.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Optional character limit fed into Unstructured (None keeps defaults).
        max_characters: Optional limit for raw extraction; pass-through to Unstructured.
        include_images: Whether to extract images into temp files.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    elements = partition_pdf(
        filename=str(pdf_path),
        extract_images_in_pdf=include_images,
        infer_table_structure=True,
        include_page_breaks=True,
        chunking_strategy="by_title" if chunk_size else None,
        chunk_size=chunk_size,
        max_characters=max_characters,
    )

    docs: List[Document] = []
    for element in elements:
        doc = element_to_document(element)
        if doc:
            docs.append(doc)
    return docs


def save_documents(documents: Iterable[Document], output_path: Path) -> None:
    """Persist parsed documents as JSON for later vectorization."""
    payload = [
        {
            "page_content": doc.page_content,
            "metadata": _ensure_json_safe(doc.metadata),
        }
        for doc in documents
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_documents(json_path: Path) -> List[Document]:
    """Utility for reloading saved documents."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return [Document(page_content=entry["page_content"], metadata=entry["metadata"]) for entry in data]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Parse a PDF into LangChain Documents.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the source PDF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/short_extract_docs.json"),
        help="Path to write JSON documents.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional chunk size hint passed to Unstructured.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=None,
        help="Optional Unstructured max_characters parameter.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Disable image extraction if storage is limited.",
    )

    args = parser.parse_args()

    documents = parse_pdf(
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        max_characters=args.max_characters,
        include_images=not args.skip_images,
    )
    save_documents(documents, args.output)
    print(f"Saved {len(documents)} documents to {args.output}")


if __name__ == "__main__":
    main()

