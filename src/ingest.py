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

try:
    import pytesseract
    from PIL import Image
except ImportError:  # OCR is optional for text-native PDFs.
    pytesseract = None
    Image = None


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


def ingest_pdf(pdf_path: str | Path, *, enable_ocr: bool = True, ocr_dpi: int = 200) -> IngestionResult:
    """Extract page text with PyMuPDF and tables with pdfplumber.

    When PyMuPDF yields no text for a page, pdfplumber and then optional OCR are
    tried as fallbacks. OCR is only run for pages that have no text, keeping
    normal text-native PDFs fast.
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

            if not raw_text and enable_ocr:
                raw_text = _ocr_page(pymupdf_page, page_number, ocr_dpi)

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


def _ocr_page(page: pymupdf.Page, page_number: int, dpi: int) -> str:
    """OCR one image-only page, returning empty text when OCR is unavailable."""

    if pytesseract is None or Image is None:
        logger.warning(
            "Page %s is image-only but OCR dependencies are not installed; "
            "install pillow and pytesseract plus the Tesseract executable.",
            page_number,
        )
        return ""
    if dpi < 72:
        raise ValueError("ocr_dpi must be at least 72")

    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        text = pytesseract.image_to_string(image)
    except Exception as exc:
        logger.warning("OCR failed for page %s: %s", page_number, exc)
        return ""
    return text.strip()


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
