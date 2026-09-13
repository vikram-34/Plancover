# PROJECT OVERVIEW

This project extracts structured data from Group Medical Cover (GMC) insurance
policy PDFs and writes one canonical JSON record per document. It is designed
to work across insurers whose layouts, headings, terminology, and table
formats differ, without relying on insurer-specific templates or conditional
extraction branches.

The output includes insurer and TPA details, policy dates and premiums, family
structure, sum-insured information, hospitalization terms, waiting periods,
maternity coverage, and other common GMC benefits. Each populated field can
also include page and snippet evidence for review.

# ARCHITECTURE / APPROACH

The extraction pipeline runs through these stages:

1. **Ingest:** `ingest.py` reads PDF text and tables with PyMuPDF (`pymupdf`),
   using `pdfplumber` as a fallback.
2. **Insurer/TPA detection:** A focused LLM call examines the opening pages
   and the final page, where closing signature blocks may identify the carrier.
3. **Grouped extraction:** The document is sent through three schema-scoped
   LLM calls: `policy_core`, `coverage_and_waiting`, and
   `benefits_and_limits`.
4. **Schedule-evidence override:** Deterministic regular expressions inspect
   labelled policy-schedule text for high-stakes values such as policy dates
   and aggregate sum insured. A clearly labelled source value takes precedence
   over an LLM value selected from a nearby table.
5. **Normalization:** Model key aliases, date formats, enum synonyms, benefit
   shorthand, and value types are normalized to the canonical field names and
   representations.
6. **Pydantic validation:** The normalized payload is validated against the
   models in `src/schema.py`.
7. **Completeness gate:** The validated record is classified as `succeeded`
   or `low_quality` based on required fields and the number of populated
   top-level fields.
8. **JSON output:** The final record is written to the configured output
   directory using the source PDF filename.

The schema is deliberately flattened for coverage fields. Instead of deeply
nested `status`/`limit`/`notes` objects, a benefit is usually represented by
one descriptive string that preserves coverage status, limits, conditions, and
source wording. This improves reliability with smaller or free-tier models.
Fields that materially benefit from structure, such as dates and demographic
counts, remain nested.

The schedule-evidence regex layer exists because an LLM can occasionally pick
the wrong value from an adjacent table. Premium-rate tables and age-band tables
are particularly likely to contain numbers that resemble the aggregate sum
insured. Deterministic overrides take precedence when the source contains a
clearly labelled policy-schedule date or aggregate sum insured.

# TECHNOLOGIES USED

- Python 3.10+
- PyMuPDF (`pymupdf`) for fast PDF text extraction
- `pdfplumber` for fallback text extraction and table reading
- Pydantic for the canonical schema and final validation
- `python-dotenv` for local configuration
- `google-genai` for the primary Gemini API backend and native structured JSON
  output
- `python-dateutil` for tolerant date parsing

Ollama with a local `llama3.1` model was evaluated during development as a
free/offline fallback option. The current configured production path uses
Gemini through `google-genai`.

# SETUP INSTRUCTIONS / INSTALLATION REQUIREMENTS

Use Python 3.10 or newer. Install the project dependencies from the repository
root:

```bash
python -m pip install -r requirements.txt
```

Copy the environment template and add a Gemini API key:

```bash
copy .env.example .env       # Windows
cp .env.example .env         # macOS/Linux
```

Create a free key at [Google AI Studio](https://aistudio.google.com). No
billing is required for the free-tier setup. Configure `.env` as follows:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.6-flash
```

`GEMINI_MODEL` is optional; the client uses its built-in default when it is not
set. Some newer preview models have very low daily quotas, sometimes around 20
requests per day. `gemini-3.6-flash` is recommended as the stable default when
it is available for the account.

## Optional Ollama development setup

Ollama was evaluated as a free, offline alternative during development. To
prepare a local Ollama environment:

```bash
ollama pull llama3.1:8b
ollama serve
```

The current checked-in pipeline is configured for Gemini; using Ollama requires
restoring or adding an Ollama client configuration rather than changing the
extraction schema or prompts.

# HOW TO RUN

To process every PDF in a folder:

```bash
python main.py <pdf_folder> --output-dir <output_folder>
```

For example:

```bash
python main.py sample_docs --output-dir output
```

The command also supports processing one PDF directly:

```bash
python main.py sample_docs/sample3.pdf --output-dir output
```

Each source document produces a file named after the PDF stem:

```text
output/sample3.json
output/GHI_Policy.json
```

The JSON `extraction_status` field is either:

- `succeeded`: the record passed validation and the completeness gate found
  the insurer, a financial term, a policy-period value, and enough populated
  fields.
- `low_quality`: a valid record was produced, but one or more completeness
  requirements were not met. Warnings explain which extraction calls or fields
  were incomplete.

One failed or malformed PDF does not stop the remaining files in a folder from
being processed.

# EXTRACTION METHODOLOGY

The extractor divides the canonical model into three calls:

- `policy_core` covers policy identity, dates, tenure, premiums, family
  structure, lives, demographics, and sum-insured tiers.
- `coverage_and_waiting` covers room rent, ICU, hospitalization periods,
  maternity, and waiting periods.
- `benefits_and_limits` covers other benefits, ambulance terms, buffers,
  infertility or surrogacy terms, and disease-wise capping.

Each group receives the full extracted document text rather than only keyword-
selected chunks. Full-document context is simpler and more reliable for the
token budget used by this project: important values often appear in schedules,
tables, footnotes, or sections whose headings do not contain the expected
keyword.

Prompts use a skeleton-fill technique. The group schema is rendered as an
exact JSON skeleton, and the model is instructed to preserve the keys while
replacing placeholders with source-backed values. Missing information is
represented deliberately rather than guessed.

The `extraction_evidence` field records a list of source page and snippet
entries for extracted fields. This makes the JSON auditable: a reviewer can
trace a premium, date, benefit, or family term back to the relevant document
text.

# JSON SCHEMA / MAPPING LOGIC

The canonical model is defined in [src/schema.py](src/schema.py). Its top-level
fields cover:

- insurer and TPA names, extraction status, and extraction evidence
- policy period, tenure, policy type, premiums, and total sum insured
- family structure, sum-insured tiers, demographic counts, and lives covered
- room rent, ICU charges, hospitalization periods, maternity, and waiting
  periods
- other benefits, infertility or surrogacy, ambulance, air ambulance,
  corporate buffer, and disease-wise capping

Most benefit and family-relation values are flattened descriptive strings. For
example, a value can preserve both a limit and its conditions:

```json
"Covered up to INR 1,000 per claim, subject to policy conditions"
```

The fields `policy_period`, `family_structure`, `demographics_counts`, and
`sum_insured_tiers` retain nested structure because dates, member relations,
counts, and tier details are easier to validate and use when structured.

The alias-mapping system uses `EXPLICIT_ALIASES` and related normalization
helpers in `extractor.py`. It maps model variants such as alternate spellings,
flattened labels, and near-match field names to canonical schema fields before
Pydantic validation. Alias usage is reported in the command output so changes
in model terminology remain visible.

# ASSUMPTIONS MADE

- Policy PDFs are expected to contain extractable text or readable tables;
  scanned image-only documents requiring OCR are outside the supported input
  assumption.
- `Not Found` is used deliberately when no direct evidence exists in the source
  text. The extractor does not infer or invent a policy term.
- Free-tier Gemini APIs are acceptable under the assignment terms; no paid API
  is required by the configured backend.
- When a date is ambiguous, dates are interpreted using the DD-MM-YYYY Indian
  date convention where the available evidence supports that interpretation.
- A PDF is treated as one policy record and produces one JSON output file.
- Monetary values, percentages, day counts, and policy wording are preserved as
  stated rather than converted into one universal unit or interpretation.

# KNOWN LIMITATIONS

- Extraction completeness varies by document. Standard GMC documents with
  detailed benefit sections typically produce about 15-16 of the 23 top-level
  fields. Sparse documents, summary or declaration pages, and non-GMC coverage
  types such as Group Personal Accident policies can correctly produce fewer
  fields because GMC concepts such as maternity, ICU, or room rent do not exist
  in the source. Fewer populated fields do not necessarily mean extraction
  failed.
- Insurer-name detection can miss cases where the insurer appears only in a
  letterhead or logo area or only in a closing signature block rather than as
  an explicit label. The detector checks opening and closing-page text, but
  visual-only logos may not be represented in extracted text.
- Smaller or free-tier LLMs occasionally need the deterministic regex-evidence
  override for critical dates and numeric fields. Pure LLM extraction can
  select a nearby figure from a similar-looking table.
- Free-tier API rate limits vary by model. Some preview models allow as few as
  approximately 20 requests per day, which constrains batch size in a single
  day.
- No OCR fallback is provided for scanned or image-only PDFs. Complex merged
  cells, footnotes, handwritten annotations, and visual symbols may also be
  extracted unreliably.
- The output is an extraction aid, not a substitute for reviewing policy
  wording, exclusions, sub-limits, and legal terms.
