# Survey Response Encoding Pipeline

## Overview

Ingests raw survey data (`.sav` or `.csv`) and outputs two deliverables:
- `encoded_response_matrix.csv` — integer matrix, one row per respondent, one column per question/sub-option, indexed by `respondent_id`
- `lookup_table.csv` — codebook mapping every encoded integer back to its question text and response label

### Important points:
- Ranked columns were not included for the following reasons:
  - example `.csv` did not contain ranked columns -> no encoding schema
  - example `.sav` did not contain ranked columns -> no encoding schema
  - `instructions.md` did not contain ranked columns -> no encoding schema
- the `natural_language_map` and `response_text_proc` columns aren't processed strictly
  - `natural_language_map` => copy of `question_text_proc`
  - `response_text_proc` => copy of `response_text`

- missing values are replaced with a sentinel value `-1` in the final encoded matrix

## Setup & Running

```bash
# 1. Clone the repo
git clone https://github.com/william-lizr/data_analyst_final.git
cd data_analyst_final

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

5. Paste your data file into the `data/` folder

6. Open `config.yaml` and set the filename:

```yaml
input_path: "data/your_file.sav"   # .sav or .csv
output_dir: "output"
validation_backcheck_rows: 50
```

7. Run the pipeline:

```bash
python pipeline.py
```

Terminal output shows a progress bar for each stage, followed by validation results and data quality flags.

---

## Assumptions

### SAV
- Column names are SPSS variable names (e.g. `var48`); human-readable question text comes from `meta.column_names_to_labels`
- Response options come from `meta.variable_value_labels`; raw numeric codes are translated to text labels before encoding
- `meta.variable_measure` takes priority over heuristics for column classification — variables with value labels or measure `nominal`/`ordinal` are never classified as `free_text`
- `apply_value_formats=False` is passed to `pyreadstat` to prevent silent coercion of strings like `"None"` or `"NA"` to NaN
- Empty strings `""` are treated as missing; all other strings (including `"None"`, `"NA"`) are treated as valid response text

### CSV
- Column headers are used as question text directly
- Multi-select columns are detected by `|` in cell values — assumed `|` never appears within a single legitimate response
- Free text detected heuristically: non-numeric string column with ≥20 unique non-null values AND ≥50% of non-null values unique
- Columns whose header contains `?` get a `Q` prefix code; all others get `A`
- `keep_default_na=False` passed to `pd.read_csv` — only genuinely empty cells become `NaN`

---

## Validation checks

Run automatically after encoding. Each prints ✅ or ❌:

| # | Check | What it verifies |
|---|---|---|
| 1 | **Participant count** | Encoded matrix row count matches source |
| 2 | **Column count** | Multi-select columns expand to one column per unique option; all others produce one column |
| 3 | **Missing value count** | Total `-1` cells matches total original `NaN`s |
| 4 | **Row-level recovery** | Random sample (default 50) fully decoded via lookup and compared to source |

---

## Data quality flags

Printed after validation:

| Flag | Threshold |
|---|---|
| Overall missing rate | >30% |
| Columns with high missingness | >50% missing |
| Respondents with high missingness | >50% of questions missing |
| Response dominance | one answer >95% of valid responses |
| Straight-lining | respondent gave identical answer to all SC questions |
| Duplicate rows | any exact duplicate response pattern |
| Lookup / matrix sync | any orphaned codes on either side |
| Question type breakdown | count per type |

---

## Pipeline steps

### 1. Load
- `.sav`: `pyreadstat.read_sav(..., apply_value_formats=False)` — builds `col_label_map`, `value_label_map`, `variable_measure` from metadata
- `.csv`: `pd.read_csv(..., keep_default_na=False, na_values=[""])`
- Exact empty strings `""` replaced with `NaN` on string columns only

### 2. Respondent IDs
UUID-based `respondent_id` (`id_<12 hex chars>`) prepended and set as index. Regenerated on each run.

### 3. Snapshot missing values
Boolean mask `original_missing` captured after empty-string replacement, before encoding. Used as the sole source of truth for `-1` placement.

### 4. Build lookup table
One loop over all columns classifies each column and appends rows to the lookup — the single source of truth for all integer ↔ text mappings.

**Column types (evaluated in order):**

| Type | Rule |
|---|---|
| `multi_select` | Any cell contains `\|` |
| `free_text` | ≥20 unique values AND ≥50% unique; SAV: excluded if value labels or `nominal`/`ordinal` measure present |
| `single_choice` | Everything else |

**Code assignment:**
- Label contains `?` → `Q` prefix (survey question)
- Otherwise → `A` prefix (demographic / metadata)
- Multi-select sub-options: `Q1_R1`, `Q1_R2`, etc.; two lookup rows each (`response_code=1` Unchecked, `response_code=2` option text)
- All-NaN columns: sentinel row (`response_code=0`) so they appear in both outputs

### 5. Encode
Three passes accumulate columns into a `dict`, then joined once with `pd.concat` (avoids fragmentation warnings):

- **Multi-select** — split on `|`; `2` if option present, `1` otherwise
- **Free text** — map unique values to sequential integers via lookup
- **Single choice** — map response text to integer codes via lookup (SAV: numeric codes translated through `value_label_map` first)

Column order: `A` ascending, then `Q` ascending.

### 6. Apply missing policy
`-1` stamped onto all cells that were originally `NaN` (per `original_missing`). Remaining `NaN` filled with `0`. Matrix cast to `int`.

| Value | Meaning |
|---|---|
| `-1` | Originally missing |
| `0` | Unmapped fallback (should not appear) |
| `1` | Unchecked / first response code |
| `≥2` | Checked / subsequent response codes |

### 7. Export
```
output/
  encoded_response_matrix.csv
  lookup_table.csv
```
