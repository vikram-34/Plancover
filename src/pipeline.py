"""Batch pipeline for extracting GMC policy PDFs into validated JSON files."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .extractor import ExtractionResult, GMCExtractor
from .ingest import IngestionResult, ingest_pdf
from .schema import GMCPPolicy


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartyDetection:
    """Insurer and TPA values found from generic labels before LLM extraction."""

    insurer_name: Optional[str] = None
    tpa_name: Optional[str] = None


@dataclass
class FileProcessResult:
    """Outcome and warnings for one source PDF."""

    pdf_path: Path
    output_path: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)
    alias_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.output_path is not None and self.error is None


@dataclass
class BatchProcessResult:
    """Aggregate outcome for a folder of policy PDFs."""

    files: list[FileProcessResult] = field(default_factory=list)
    alias_mappings: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def succeeded_count(self) -> int:
        return sum(result.succeeded for result in self.files)

    @property
    def failed_count(self) -> int:
        return len(self.files) - self.succeeded_count


def process_folder(
    pdf_folder: str | Path,
    output_folder: str | Path,
    *,
    extractor: GMCExtractor | None = None,
) -> BatchProcessResult:
    """Process every PDF in a folder, continuing when an individual file fails."""

    source_dir = Path(pdf_folder)
    destination_dir = Path(output_folder)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"PDF folder not found: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        (path for path in source_dir.iterdir() if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )
    active_extractor = extractor or GMCExtractor()
    batch = BatchProcessResult()

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{index}/{len(pdf_files)}] Processing {pdf_path.name}")
        result = process_pdf(pdf_path, destination_dir, active_extractor)
        batch.files.append(result)
        _merge_alias_usage(batch.alias_mappings, result.alias_mappings)
        _print_file_result(result)

    return batch


def process_pdf(pdf_path: Path, output_folder: Path, extractor: GMCExtractor) -> FileProcessResult:
    """Ingest, detect parties, extract, validate, and save one policy PDF."""

    result = FileProcessResult(pdf_path=pdf_path)
    try:
        document = ingest_pdf(pdf_path)
        detected_parties = detect_insurer_and_tpa(document)
        extraction = extractor.extract(document)
        result.warnings.extend(extraction.errors)
        result.alias_mappings = extraction.alias_mappings

        policy = _merge_detected_parties(extraction, detected_parties)
        if policy is None:
            raise ValueError("No valid policy JSON was produced; see validation warnings.")

        output_path = output_folder / f"{pdf_path.stem}.json"
        output_path.write_text(policy.model_dump_json(indent=2), encoding="utf-8")
        result.output_path = output_path
    except Exception as exc:
        result.error = str(exc)
        logger.error("Failed to process %s: %s", pdf_path.name, exc)

    return result


def detect_insurer_and_tpa(document: IngestionResult) -> PartyDetection:
    """Find labelled insurer/TPA values without insurer-specific name lists.

    This inexpensive first pass only accepts explicit labels such as ``Insurer``
    or ``Third Party Administrator``.  The LLM extraction remains responsible
    for documents whose layout does not expose those labels clearly.
    """

    text = "\n".join(page.raw_text for page in document.pages)
    return PartyDetection(
        insurer_name=_find_labelled_value(
            text,
            ("insurer", "insurance company", "underwriter"),
        ),
        tpa_name=_find_labelled_value(
            text,
            ("tpa", "third party administrator", "third-party administrator"),
        ),
    )


def _merge_detected_parties(
    extraction: ExtractionResult, detected: PartyDetection
) -> Optional[GMCPPolicy]:
    """Fill only missing party fields, then validate the final merged record."""

    if extraction.policy is None:
        return None

    payload = extraction.policy.model_dump(mode="json")
    if payload.get("insurer_name") is None:
        payload["insurer_name"] = detected.insurer_name
    if payload.get("tpa_name") is None:
        payload["tpa_name"] = detected.tpa_name

    try:
        return GMCPPolicy.model_validate(payload)
    except ValidationError as exc:
        message = f"Validation after party detection failed: {exc}"
        logger.error(message)
        extraction.errors.append(message)
        return None


def _find_labelled_value(text: str, labels: tuple[str, ...]) -> Optional[str]:
    """Extract one short line value adjacent to a case-insensitive field label."""

    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?im)^\s*(?:{label_pattern})\s*(?:name)?\s*[:\-–]\s*(.+?)\s*$",
        text,
    )
    if not match:
        return None

    value = re.sub(r"\s+", " ", match.group(1)).strip(" -:–")
    return value or None


def _print_file_result(result: FileProcessResult) -> None:
    if result.succeeded:
        print(f"  Saved: {result.output_path}")
    else:
        print(f"  Failed: {result.error}")
    for warning in result.warnings:
        print(f"  Warning: {warning}")


def _merge_alias_usage(
    destination: dict[str, dict[str, int]], source: dict[str, dict[str, int]]
) -> None:
    for source_key, targets in source.items():
        destination_targets = destination.setdefault(source_key, {})
        for target, count in targets.items():
            destination_targets[target] = destination_targets.get(target, 0) + count
