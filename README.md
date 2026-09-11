# GMC Policy Extraction

An AI-assisted Python tool for extracting structured data from Group Medical
Cover (GMC) insurance policy PDFs. It is intended for policies from different
insurers, whose layouts, headings, terminology, and table formats vary widely.

## Architecture and approach

The pipeline is:

1. `ingest.py` reads every page with PyMuPDF, falls back to pdfplumber when a
   page has no extractable text, and extracts page-level tables.
2. `pipeline.py` discovers PDFs, performs a lightweight generic insurer/TPA
   label detection pass, invokes extraction, validates the result, and writes
   one JSON file per PDF.
3. `extractor.py` uses the Pydantic model as the source of truth for the output
   structure. The document is sent to the local LLM with a complete JSON
   skeleton, followed by one repair call if parsing or validation fails.
4. The normalization layer handles common date formats, enum synonyms, benefit
   shorthand, and field-name aliases before final Pydantic validation.

The design is schema-driven and insurer-agnostic. Concept-based chunking and
keyword routing helpers are available for logical page grouping and targeted
prompt extraction when needed; the current fast path uses one full-document
prompt to reduce the number of LLM calls and avoid losing information scattered
across pages. No insurer-specific template or conditional branch is required.

## Technologies

- Python 3.10+
- Pydantic for the canonical output model and validation
- PyMuPDF (`pymupdf`) for fast page text extraction
- pdfplumber for fallback text extraction and tables
- Ollama through its OpenAI-compatible API
- `python-dotenv` for local configuration
- `python-dateutil` for tolerant policy-date parsing

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama, start its local server, and pull the model you want to use:

```bash
ollama pull llama3.1:8b
ollama serve
```

Copy the environment template and optionally change the model:

```bash
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
```

Example `.env`:

```env
OLLAMA_MODEL=llama3.1:8b
```

The application connects to `http://localhost:11434/v1`. The model must already
be available locally; no cloud API key is required.

## Running the extractor

Place one or more policy PDFs in `sample_docs/`, then run:

```bash
python main.py
```

Use custom source and output folders when needed:

```bash
python main.py path/to/pdfs --output-dir path/to/json
```

The command prints per-file progress, warnings, success/failure counts, and the
field aliases used during the run. Results are written as:

```text
output/<pdf-filename-without-extension>.json
```

One malformed or failed PDF does not stop the remaining files from processing.

## JSON schema

The canonical model is defined in [src/schema.py](src/schema.py). It captures:

- insurer, TPA, policy period, tenure, and premiums
- family structure, sum-insured tiers, demographics, and lives covered
- room-rent, ICU, and hospitalization-period terms
- maternity limits and newborn/vaccination cover
- initial, first-year, second-year, and pre-existing-disease waiting periods
- common and special benefits such as OPD, daycare, AYUSH, ambulance, buffer,
  organ donor, psychiatric, bariatric, and modern treatment cover

Coverage-like fields use a consistent structure:

```json
{
  "status": "Covered",
  "limit": "INR 1,000 per claim",
  "notes": "Subject to policy conditions"
}
```

Valid benefit statuses are `Covered`, `Not Covered`, and `Waived Off`. Waiting
periods use `Applied`, `Not Covered`, and `Waived Off`. Monetary amounts,
percentages, and day counts are kept as strings where preserving the source
wording is important.

## Assumptions

- The PDF contains selectable text or tables for the relevant information.
- A policy document represents one policy record and produces one JSON file.
- Missing information is represented as `null` or an empty list after the
  model's explicit `Not Found` responses are normalized.
- Values are preserved as stated rather than converted into a single currency,
  unit, or interpretation.
- Ollama is available locally at the configured OpenAI-compatible endpoint.

## Known limitations

- Extraction quality depends on the local model's ability to follow the exact
  JSON skeleton and interpret ambiguous insurance phrasing.
- A full-document call is efficient in request count but can be affected by
  model context limits on very long policies.
- Scanned or image-only pages do not receive OCR; pdfplumber fallback may still
  return no text. OCR quality and table reading can vary substantially by scan
  quality and PDF layout.
- Complex merged tables, footnotes, handwritten annotations, and visual symbols
  may not be extracted reliably.
- Alias normalization handles known, unambiguous variants, but genuinely new
  insurer terminology may be logged and omitted until a schema field or alias is
  added.
- The output is an extraction aid, not a substitute for human review of policy
  wording, exclusions, sub-limits, and legal terms.

## Final results

Five out of five sample policies—`GHI_Policy.pdf` and `sample1.pdf` through
`sample4.pdf`—were processed successfully end-to-end with no pipeline failures.

Known extraction limitations from the sample run:

- For some insurers' phrasing, `other_benefits` occasionally comes back as a
  flat list of key-value pairs instead of the nested canonical schema. This is
  logged as a warning and gracefully skipped rather than crashing the pipeline.
- A handful of insurer-specific fields outside the canonical schema, such as
  `Maternity_Claims`, `Proposal_Acceptance`, and `Policy_Details`, are similarly
  logged as unexpected fields and skipped; they are not silently dropped.
- The solution deliberately uses the local Ollama model `llama3.1:8b` rather
  than a paid API because of resource constraints. This saves API cost and
  keeps policy data local, but is a tradeoff: nested-field extraction can be
  less consistent than with frontier API models.

## Tests

Basic tests are in `tests/` and cover schema validation, sample PDF ingestion,
key normalization, and no-information handling. Run them after installing
pytest:

```bash
pip install pytest
python -m pytest -q
```
