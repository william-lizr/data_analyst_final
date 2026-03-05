import pandas as pd
import numpy as np
import pyreadstat
import os
import re
import uuid
import random
import yaml
from tqdm import tqdm

############################################################
# CONFIG
############################################################

with open("config.yaml", "r") as _f:
    _cfg = yaml.safe_load(_f)

input_path                = _cfg["input_path"]
output_dir                = _cfg.get("output_dir", "output")
validation_backcheck_rows = _cfg.get("validation_backcheck_rows", 50)

os.makedirs(output_dir, exist_ok=True)

############################################################
# LOAD FILE
############################################################

if input_path.endswith(".sav"):

    df, meta = pyreadstat.read_sav(
        input_path,
        apply_value_formats=False,
    )
else:

    df = pd.read_csv(input_path, keep_default_na=False, na_values=[""])
    meta = None

df.columns = df.columns.str.strip()

_str_cols = df.select_dtypes(include=["object", "str"]).columns
for _col in tqdm(_str_cols, desc="Cleaning empty strings", unit="col"):
    df[_col] = df[_col].replace("", np.nan)

def build_label_map(columns, meta):
    if meta is not None and hasattr(meta, "column_names_to_labels"):
        return {
            col: (meta.column_names_to_labels.get(col) or col).strip()
            for col in columns
        }
    return {col: col for col in columns}

col_label_map = build_label_map(df.columns, meta)

label_to_col_map = {v: k for k, v in col_label_map.items()}
def build_value_label_map(meta):
    if meta is not None and hasattr(meta, "variable_value_labels"):
        
        return {
            var: {str(k): str(v) for k, v in labels.items()}
            for var, labels in meta.variable_value_labels.items()
        }
    return {}

value_label_map = build_value_label_map(meta)

############################################################
# RESPONDENT ID
############################################################

df.insert(0, "respondent_id", ["id_" + uuid.uuid4().hex[:12] for _ in range(len(df))])
df = df.set_index("respondent_id")

############################################################
# TRACK ORIGINAL MISSING
############################################################

original_missing = df.isna()

############################################################
# META-DRIVEN COLUMN CLASSIFICATION
############################################################

def build_measure_map(meta):
    if meta is not None and hasattr(meta, "variable_measure"):
        return dict(meta.variable_measure)  
    return {}

variable_measure = build_measure_map(meta)

############################################################
# DETECTION HELPERS
############################################################

FREE_TEXT_UNIQUE_THRESHOLD = 0.5  # proportion of unique non-null string values above which = free_text
FREE_TEXT_MIN_UNIQUE = 20         # minimum absolute unique count to qualify as free_text

def is_free_text(col_name, series):

    if col_name in value_label_map and value_label_map[col_name]:
        return False

    measure = variable_measure.get(col_name, "")
    if measure in ("nominal", "ordinal"):
        return False

    non_null = series.dropna().astype(str)
    if pd.api.types.is_numeric_dtype(series.dropna()):
        return False
    # exclude multi-select
    if non_null.str.contains(r"\|").any():
        return False
    n_unique = non_null.nunique()
    n_total  = len(non_null)
    if n_total == 0:
        return False
    return n_unique >= FREE_TEXT_MIN_UNIQUE and (n_unique / n_total) >= FREE_TEXT_UNIQUE_THRESHOLD

############################################################
# BUILD LOOKUP
############################################################

lookup_rows = []

q_counter = 1
a_counter = 1

for col in tqdm(df.columns, desc="Building lookup", unit="col"):

    if col == "respondent_id":
        continue

    values = df[col].dropna().astype(str)
    question_text_proc = col_label_map.get(col, col).strip()

    col_value_labels = value_label_map.get(col, {})
    if col_value_labels:
        values = values.map(lambda v: col_value_labels.get(v, v))

    ########################################################
    # BUILD PARENT QUESTION CODE
    ########################################################

    if "?" in question_text_proc:
        parent_code = f"Q{q_counter}"
        q_counter += 1
    else:
        parent_code = f"A{a_counter}"
        a_counter += 1

    ########################################################
    # DETECT TYPE
    ########################################################

    if values.str.contains(r"\|").any():
        qtype = "multi_select"
    elif is_free_text(col, df[col]):
        qtype = "free_text"
    else:
        qtype = "single_choice"

    ########################################################
    # MULTISELECT
    ########################################################

    if qtype == "multi_select":

        options = (
            values
            .str.split("|")
            .explode()
            .str.strip()
            .dropna()
            .unique()
        )

        for i, opt in enumerate(sorted(options), start=1):

            question_code = f"{parent_code}_R{i}"

            lookup_rows.append({
                "question_text":        f"{question_code} {question_text_proc}",
                "question_text_proc":   question_text_proc,
                "question_code":        question_code,
                "parent_question_code": parent_code,
                "question_type":        qtype,
                "response_code":        1,
                "response_text":        "Unchecked",
                "natural_language_map": question_text_proc,
                "response_text_proc":   "Unchecked",
                "source_varname":       col,
            })

            lookup_rows.append({
                "question_text":        f"{question_code} {question_text_proc}",
                "question_text_proc":   question_text_proc,
                "question_code":        question_code,
                "parent_question_code": parent_code,
                "question_type":        qtype,
                "response_code":        2,
                "response_text":        opt,
                "natural_language_map": question_text_proc,
                "response_text_proc":   opt,
                "source_varname":       col,
            })

    ########################################################
    # FREE TEXT — one lookup row per unique value, single encoded column
    ########################################################

    elif qtype == "free_text":

        unique_vals = values.unique()

        if len(unique_vals) == 0:
            lookup_rows.append({
                "question_text":        f"{parent_code} {question_text_proc}",
                "question_text_proc":   question_text_proc,
                "question_code":        parent_code,
                "parent_question_code": parent_code,
                "question_type":        qtype,
                "response_code":        0,
                "response_text":        "",
                "natural_language_map": question_text_proc,
                "response_text_proc":   "",
                "source_varname":       col,
            })
        else:
            for i, opt in enumerate(sorted(unique_vals), start=1):

                lookup_rows.append({
                    "question_text":        f"{parent_code} {question_text_proc}",
                    "question_text_proc":   question_text_proc,
                    "question_code":        parent_code,
                    "parent_question_code": parent_code,
                    "question_type":        qtype,
                    "response_code":        i,
                    "response_text":        opt,
                    "natural_language_map": question_text_proc,
                    "response_text_proc":   opt,
                    "source_varname":       col,
                })

    ########################################################
    # SINGLE CHOICE
    ########################################################

    else:

        options = values.unique()

        if len(options) == 0:
            lookup_rows.append({
                "question_text":        f"{parent_code} {question_text_proc}",
                "question_text_proc":   question_text_proc,
                "question_code":        parent_code,
                "parent_question_code": parent_code,
                "question_type":        qtype,
                "response_code":        0,
                "response_text":        "",
                "natural_language_map": question_text_proc,
                "response_text_proc":   "",
                "source_varname":       col,
            })
        else:
            for i, opt in enumerate(sorted(options), start=1):

                lookup_rows.append({
                    "question_text":        f"{parent_code} {question_text_proc}",
                    "question_text_proc":   question_text_proc,
                    "question_code":        parent_code,
                    "parent_question_code": parent_code,
                    "question_type":        qtype,
                    "response_code":        i,
                    "response_text":        opt,
                    "natural_language_map": question_text_proc,
                    "response_text_proc":   opt,
                    "source_varname":       col,
                })

lookup = pd.DataFrame(lookup_rows)

############################################################
# BUILD ENCODED MATRIX
############################################################

encoded_cols = {}

############################################################
# MULTISELECT EXPANSION
############################################################

multi_questions = lookup.loc[
    lookup.question_type == "multi_select",
    "parent_question_code"
].unique()

for parent in tqdm(multi_questions, desc="Encoding multi-select", unit="q"):

    source_varname = lookup.loc[
        lookup.parent_question_code == parent,
        "source_varname"
    ].iloc[0]

    parent_lookup = lookup.loc[
        (lookup.parent_question_code == parent) & (lookup.response_code == 2)
    ].sort_values("question_code")

    split_vals = df[source_varname].str.split("|")  # NaN stays NaN

    for _, lrow in parent_lookup.iterrows():
        qcode = lrow["question_code"]
        option_text = lrow["response_text_proc"]

        encoded_cols[qcode] = split_vals.apply(
            lambda x: 2 if isinstance(x, list) and option_text in [v.strip() for v in x] else 1
        )

############################################################
# FREE TEXT ENCODING  (integer code per unique value, single column)
############################################################

free_text_questions = lookup.loc[
    lookup.question_type == "free_text",
    "parent_question_code"
].unique()

for parent in free_text_questions:

    source_varname = lookup.loc[
        lookup.parent_question_code == parent,
        "source_varname"
    ].iloc[0]

    mapping = dict(
        zip(
            lookup.loc[lookup.parent_question_code == parent]["response_text"],
            lookup.loc[lookup.parent_question_code == parent]["response_code"]
        )
    )

    col_val_labels = value_label_map.get(source_varname, {})
    str_col = df[source_varname].where(
        df[source_varname].isna(),
        df[source_varname].astype(str).map(lambda v: col_val_labels.get(v, v))
    )
    encoded_cols[parent] = str_col.map(mapping)

############################################################
# SINGLE CHOICE ENCODING
############################################################

single_questions = lookup.loc[
    lookup.question_type == "single_choice",
    "parent_question_code"
].unique()

for parent in tqdm(single_questions, desc="Encoding single-choice", unit="q"):

    source_varname = lookup.loc[
        lookup.parent_question_code == parent,
        "source_varname"
    ].iloc[0]

    if source_varname not in df.columns:
        continue

    mapping = dict(
        zip(
            lookup.loc[lookup.parent_question_code == parent]["response_text"],
            lookup.loc[lookup.parent_question_code == parent]["response_code"]
        )
    )
    col_val_labels = value_label_map.get(source_varname, {})
    str_col = df[source_varname].where(
        df[source_varname].isna(),
        df[source_varname].astype(str).map(lambda v: col_val_labels.get(v, v))
    )
    encoded_cols[parent] = str_col.map(mapping)

############################################################
# CONCAT + CONVERT TO NUMERIC
############################################################

encoded = pd.concat(encoded_cols, axis=1)
encoded = encoded.apply(pd.to_numeric, errors="coerce")

############################################################
# APPLY MISSING POLICY
############################################################

col_to_source_varname = {}
for enc_col in encoded.columns:
    match = lookup.loc[lookup.question_code == enc_col, "source_varname"]
    if not match.empty:
        col_to_source_varname[enc_col] = match.iloc[0]

for enc_col in tqdm(encoded.columns, desc="Applying missing policy", unit="col"):
    source_col = col_to_source_varname.get(enc_col)
    if source_col and source_col in original_missing.columns:
        encoded.loc[original_missing[source_col], enc_col] = -1

encoded = encoded.fillna(0).astype(int)

############################################################
# COLUMN ORDER
############################################################

def sort_key(col):
    m = re.match(r"([AQ])(\d+)", col)
    if m:
        letter = m.group(1)
        number = int(m.group(2))
        letter_order = 0 if letter == "A" else 1
        return (letter_order, number, col)
    else:
        return (2, 0, col)

encoded = encoded.reindex(sorted(encoded.columns, key=sort_key), axis=1)

############################################################
# SCHEMA VALIDATION
############################################################

required_cols = [
    "question_text",
    "question_text_proc",
    "question_code",
    "parent_question_code",
    "question_type",
    "response_code",
    "response_text",
    "natural_language_map",
    "response_text_proc",
    "source_varname",   # new required column
]

missing_cols = [c for c in required_cols if c not in lookup.columns]

if missing_cols:
    raise ValueError(f"Lookup missing columns: {missing_cols}")

# Report how much of the SAV meta was consumed
if meta is not None:
    n_cols = len(df.columns)
    n_labelled   = sum(1 for c in df.columns if c in value_label_map and value_label_map[c])
    n_measured   = sum(1 for c in df.columns if variable_measure.get(c, "") in ("nominal", "ordinal", "scale"))
    n_col_labels = sum(1 for c in df.columns if col_label_map.get(c, c) != c)
    print(f"\n=== SAV META COVERAGE ===")
    print(f"  Columns with human labels (column_names_to_labels): {n_col_labels}/{n_cols}")
    print(f"  Columns with value labels (variable_value_labels):  {n_labelled}/{n_cols}")
    print(f"  Columns with measure type (variable_measure):       {n_measured}/{n_cols}")

if encoded.index.name != "respondent_id":
    raise ValueError("Index must be respondent_id")

if encoded.select_dtypes(exclude="int").shape[1] > 0:
    raise ValueError("Encoded matrix must contain only integers")

if encoded.isna().sum().sum() > 0:
    raise ValueError("NaNs present in encoded matrix")

############################################################
# VALIDATION
############################################################

print("\n=== VALIDATION ===")
all_checks_passed = True

def fail(msg):
    global all_checks_passed
    all_checks_passed = False
    print(f"  ❌ FAIL: {msg}")

def ok(msg):
    print(f"  ✅ OK: {msg}")

# ----------------------------------------------------------
# CHECK 1: Number of participants
# ----------------------------------------------------------

if len(df) == len(encoded):
    ok(f"Participant count matches: {len(df)}")
else:
    fail(f"Participant count mismatch: df={len(df)}, encoded={len(encoded)}")

# ----------------------------------------------------------
# CHECK 2: Column count
# ----------------------------------------------------------

expected_encoded_cols = 0
for col in df.columns:
    values = df[col].dropna().astype(str)
    if values.str.contains(r"\|").any():
        n_options = values.str.split("|").explode().str.strip().dropna().unique()
        expected_encoded_cols += len(n_options)
    else:
        expected_encoded_cols += 1  # single_choice, free_text, single_value = 1 column

if len(encoded.columns) == expected_encoded_cols:
    ok(f"Column count matches: {len(encoded.columns)}")
else:
    fail(f"Column count mismatch: expected={expected_encoded_cols}, encoded={len(encoded.columns)}")

# ----------------------------------------------------------
# CHECK 3: Missing values (-1 codes)
# ----------------------------------------------------------

expected_minus1 = 0
for col in df.columns:
    n_missing = original_missing[col].sum()   # use snapshot (includes "" → NaN)
    if n_missing == 0:
        continue
    if df[col].dropna().astype(str).str.contains(r"\|").any():
        n_subcols = lookup.loc[
            (lookup.source_varname == col) & (lookup.response_code == 2),
            "question_code"
        ].nunique()
        expected_minus1 += n_missing * n_subcols
    else:
        # single_choice and free_text both produce one encoded column
        expected_minus1 += n_missing

encoded_missing_count = (encoded == -1).sum().sum()

if expected_minus1 == encoded_missing_count:
    ok(f"Missing value count matches: {encoded_missing_count} (-1 codes)")
else:
    fail(f"Missing value count mismatch: expected={expected_minus1}, encoded_(-1)={encoded_missing_count}")

# ----------------------------------------------------------
# CHECK 4: Row-level recovery check on random sample
# ----------------------------------------------------------

sample_ids = random.sample(list(df.index), min(validation_backcheck_rows, len(df)))
print(f"\n  Row-level recovery check (sample={len(sample_ids)}):")

for rid in tqdm(sample_ids, desc="Row recovery check", unit="row"):
    original_row = df.loc[rid]
    encoded_row  = encoded.loc[rid]
    errors = []

    for col in df.columns:
        orig_val = original_row[col]

        # Translate raw SAV float value to its label for comparison
        col_val_labels = value_label_map.get(col, {})
        orig_val_str = col_val_labels.get(str(orig_val), str(orig_val)) if not pd.isna(orig_val) else orig_val

        if pd.isna(orig_val):
            q_codes = lookup.loc[lookup.source_varname == col, "question_code"].unique()
            for qc in q_codes:
                if qc in encoded_row.index and encoded_row[qc] != -1:
                    errors.append(f"{col}: expected -1 for missing, got {encoded_row[qc]}")
            continue

        col_label = col_label_map.get(col, col)

        values = df[col].dropna().astype(str)
        is_multi = values.str.contains(r"\|").any()

        if is_multi:
            checked_options = []
            sub = lookup.loc[
                (lookup.source_varname == col) & (lookup.response_code == 2)
            ]
            for _, lrow in sub.iterrows():
                qc = lrow["question_code"]
                if qc in encoded_row.index and encoded_row[qc] == 2:
                    checked_options.append(lrow["response_text_proc"])

            original_options = sorted([v.strip() for v in str(orig_val).split("|")])
            recovered_options = sorted(checked_options)

            if original_options != recovered_options:
                errors.append(f"{col} ({col_label}): original={original_options}, recovered={recovered_options}")

        else:
            q_code_rows = lookup.loc[
                (lookup.source_varname == col) &
                (lookup.question_type.isin(["single_choice", "free_text", "single_value"])),
                "question_code"
            ]
            if q_code_rows.empty:
                continue
            q_code = q_code_rows.iloc[0]

            if q_code in encoded_row.index:
                enc_code = encoded_row[q_code]
                recovered = lookup.loc[
                    (lookup.question_code == q_code) & (lookup.response_code == enc_code),
                    "response_text_proc"
                ]
                recovered_val = recovered.iloc[0] if not recovered.empty else None
                if str(recovered_val) != orig_val_str:
                    errors.append(f"{col} ({col_label}): original='{orig_val_str}', recovered='{recovered_val}'")

    if errors:
        fail(f"Row {rid}:")
        for e in errors:
            print(f"       {e}")
    else:
        ok(f"Row {rid} recovered correctly")

# ----------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------

print()
if all_checks_passed:
    print("✅ All validation checks passed.")
else:
    print("❌ Some validation checks failed — review output above.")


############################################################
# DATA QUALITY REPORT
############################################################

print("\n=== DATA QUALITY REPORT ===")

n_respondents = len(encoded)
n_questions   = len(encoded.columns)
total_cells   = n_respondents * n_questions

# ----------------------------------------------------------
# 1. Overall missing rate
# ----------------------------------------------------------

n_missing_cells = (encoded == -1).sum().sum()
pct_missing_overall = 100 * n_missing_cells / total_cells
print(f"\n  Missingness")
print(f"  {'Overall missing rate:':<45} {pct_missing_overall:.2f}%  ({n_missing_cells:,} / {total_cells:,} cells)")

# Flag if overall missingness is unusually high
if pct_missing_overall > 30:
    print(f"  ⚠️  WARNING: overall missing rate exceeds 30%")

# Columns with very high missingness (>50%)
col_missing_pct = (encoded == -1).mean() * 100
high_missing_cols = col_missing_pct[col_missing_pct > 50].sort_values(ascending=False)
if not high_missing_cols.empty:
    print(f"\n  ⚠️  Columns with >50% missing ({len(high_missing_cols)}):")
    for enc_col, pct in high_missing_cols.items():
        src = lookup.loc[lookup.question_code == enc_col, "source_varname"]
        src_str = src.iloc[0] if not src.empty else "?"
        print(f"       {enc_col:<12} ({src_str}): {pct:.1f}% missing")

# Respondents with very high missingness (>50% of their questions missing)
row_missing_pct = (encoded == -1).mean(axis=1) * 100
high_missing_rows = row_missing_pct[row_missing_pct > 50]
print(f"\n  Respondents with >50% questions missing: {len(high_missing_rows)} / {n_respondents}")
if not high_missing_rows.empty:
    print(f"  ⚠️  WARNING: {len(high_missing_rows)} respondents are mostly missing")

# Respondents with 100% missing (entirely blank submissions)
all_missing_rows = row_missing_pct[row_missing_pct == 100]
if not all_missing_rows.empty:
    print(f"  ⚠️  WARNING: {len(all_missing_rows)} respondents have ALL questions missing (ghost rows)")

# ----------------------------------------------------------
# 2. Response distribution red flags
# ----------------------------------------------------------

print(f"\n  Single-choice response distributions")

# For single_choice columns: flag any where one response code dominates (>95%)
sc_cols = lookup.loc[lookup.question_type == "single_choice", "question_code"].unique()
flatline_cols = []
for enc_col in sc_cols:
    if enc_col not in encoded.columns:
        continue
    col_vals = encoded[enc_col]
    valid = col_vals[col_vals != -1]
    if len(valid) == 0:
        continue
    top_pct = valid.value_counts(normalize=True).iloc[0] * 100
    if top_pct > 95:
        top_val = valid.value_counts().index[0]
        top_label_rows = lookup.loc[
            (lookup.question_code == enc_col) & (lookup.response_code == top_val),
            "response_text_proc"
        ]
        top_label = top_label_rows.iloc[0] if not top_label_rows.empty else str(top_val)
        flatline_cols.append((enc_col, top_pct, top_label))

if flatline_cols:
    print(f"  ⚠️  Columns where one response accounts for >95% of valid answers ({len(flatline_cols)}):")
    for enc_col, pct, label in flatline_cols:
        print(f"       {enc_col:<12}: {pct:.1f}% answered '{label}'")
else:
    print(f"  ✅  No single-choice column is dominated (>95%) by one response")

# ----------------------------------------------------------
# 3. Straight-lining detection (single_choice only)
# ----------------------------------------------------------

print(f"\n  Straight-lining (respondents giving identical answers across all single-choice questions)")

sc_encoded = encoded[[c for c in sc_cols if c in encoded.columns]]

valid_sc_mask = (sc_encoded != -1)
valid_counts  = valid_sc_mask.sum(axis=1)
candidate_rows = sc_encoded[valid_counts >= 5]
valid_mask_cand = valid_sc_mask[valid_counts >= 5]

def is_straightline(row, mask):
    
    bool_mask = mask.loc[row.name]
    vals = row[bool_mask]
    if len(vals) < 5:
        return False
    return vals.nunique() == 1

straightliners = candidate_rows.apply(lambda r: is_straightline(r, valid_mask_cand), axis=1)
n_straight = straightliners.sum()
pct_straight = 100 * n_straight / n_respondents
print(f"  Straight-liners: {n_straight} / {n_respondents}  ({pct_straight:.1f}%)")
if pct_straight > 5:
    print(f"  ⚠️  WARNING: >{pct_straight:.1f}% of respondents gave the same answer to every SC question")

# ----------------------------------------------------------
# 4. Duplicate respondents
# ----------------------------------------------------------

print(f"\n  Duplicate detection")

n_dupes = encoded.duplicated().sum()
pct_dupes = 100 * n_dupes / n_respondents
print(f"  Exact duplicate rows: {n_dupes} / {n_respondents}  ({pct_dupes:.1f}%)")
if n_dupes > 0:
    print(f"  ⚠️  WARNING: {n_dupes} respondents have identical response patterns")

# ----------------------------------------------------------
# 5. Lookup table integrity
# ----------------------------------------------------------

print(f"\n  Lookup table")
n_lookup_cols   = lookup["source_varname"].nunique()
n_encoded_cols  = len(encoded.columns)
n_orphan_enc    = sum(1 for c in encoded.columns if c not in lookup["question_code"].values)
n_orphan_lookup = lookup.loc[~lookup["question_code"].isin(encoded.columns), "question_code"].nunique()

print(f"  Source variables in lookup:   {n_lookup_cols}")
print(f"  Encoded columns:              {n_encoded_cols}")
if n_orphan_enc:
    print(f"  ⚠️  Encoded columns with no lookup entry: {n_orphan_enc}")
if n_orphan_lookup:
    print(f"  ⚠️  Lookup entries with no encoded column: {n_orphan_lookup}")
if n_orphan_enc == 0 and n_orphan_lookup == 0:
    print(f"  ✅  Lookup and encoded matrix are in sync")

# ----------------------------------------------------------
# 6. Question type breakdown
# ----------------------------------------------------------

print(f"\n  Question type breakdown (source columns)")
type_counts = (
    lookup.drop_duplicates("source_varname")
    ["question_type"].value_counts()
)
for qtype, cnt in type_counts.items():
    print(f"       {qtype:<20}: {cnt}")

print()
############################################################
# EXPORT
############################################################

encoded.to_csv(os.path.join(output_dir, "encoded_response_matrix.csv"))
lookup.to_csv(os.path.join(output_dir, "lookup_table.csv"), index=False)

print("\nPipeline completed")
print("Encoded shape:", encoded.shape)
print("Lookup rows:", len(lookup))

print(f'Output at: {output_dir}')
print(f'response_matrix at: {os.path.join(output_dir, "encoded_response_matrix.csv")}')
print(f'lookup_table: {os.path.join(output_dir, "lookup_table.csv")}')
