"""PDF text and table ingestion for GMC policy documents.

Run as ``python -m src.ingest path/to/policy.pdf`` to inspect an input document.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf
import pdfplumber


logger = logging.getLogger(__name__)

Table = list[list[Optional[str]]]


@dataclass
class PageContent:
    """Raw content extracted from one PDF page."""

    page_number: int
    raw_text: str
    tables: list[Table]


@dataclass
class IngestionResult:
    """Structured content extracted from a complete PDF document."""

    pages: list[PageContent]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_with_tables(self) -> list[int]:
        return [page.page_number for page in self.pages if page.tables]

    @property
    def total_characters(self) -> int:
        return sum(len(page.raw_text) for page in self.pages)


def ingest_pdf(pdf_path: str | Path) -> IngestionResult:
    """Extract page text with PyMuPDF and tables with pdfplumber.

    When PyMuPDF yields no text for a page, pdfplumber is tried as a fallback.
    A truly image-only/scanned page will still need a later OCR step; it is
    returned with empty text so callers can detect it without losing its tables.
    """

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    pages: list[PageContent] = []
    with pymupdf.open(path) as pymupdf_document, pdfplumber.open(path) as plumber_document:
        if len(pymupdf_document) != len(plumber_document.pages):
            logger.warning(
                "Page-count mismatch: PyMuPDF=%s, pdfplumber=%s",
                len(pymupdf_document),
                len(plumber_document.pages),
            )

        for page_index, pymupdf_page in enumerate(pymupdf_document):
            page_number = page_index + 1
            raw_text = pymupdf_page.get_text("text").strip()
            plumber_page = plumber_document.pages[page_index]

            if not raw_text:
                logger.info("Page %s has no PyMuPDF text; trying pdfplumber fallback.", page_number)
                raw_text = (plumber_page.extract_text() or "").strip()

            tables: list[Table] = plumber_page.extract_tables() or []
            if tables:
                logger.info("Page %s contains %s table(s).", page_number, len(tables))

            pages.append(
                PageContent(
                    page_number=page_number,
                    raw_text=raw_text,
                    tables=tables,
                )
            )

    return IngestionResult(pages=pages)


def main() -> None:
    """Run the PDF ingestion CLI and print a compact extraction summary."""

    parser = argparse.ArgumentParser(description="Extract text and tables from a GMC policy PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to ingest.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = ingest_pdf(args.pdf_path)
    table_pages = ", ".join(map(str, result.pages_with_tables)) or "none"

    print(f"Page count: {result.page_count}")
    print(f"Pages with tables: {table_pages}")
    print(f"Total characters extracted: {result.total_characters}")


if __name__ == "__main__":
    main()
