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

input_path = _cfg["input_path"]
output_dir = _cfg.get("output_dir", "output")
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
FREE_TEXT_MIN_UNIQUE = 20  # minimum absolute unique count to qualify as free_text


def is_free_text(col_name, series):
    if col_name in value_label_map and value_label_map[col_name]:
        return False

    measure = variable_measure.get(col_name, "")
    if measure in ("nominal", "ordinal"):
        return False

    non_null = series.dropna().astype(str)
    if pd.api.types.is_numeric_dtype(series.dropna()):
        return False
    # exclude pipe-delimited multi-select (CSV path)
    if non_null.str.contains(r"\|").any():
        return False
    n_unique = non_null.nunique()
    n_total = len(non_null)
    if n_total == 0:
        return False
    return n_unique >= FREE_TEXT_MIN_UNIQUE and (n_unique / n_total) >= FREE_TEXT_UNIQUE_THRESHOLD


############################################################
# MULTI-SELECT GROUP DETECTION  (SAV-specific)
#
# Returns a dict:  { parent_label: [col1, col2, ...], ... }
# where every col in the list is a binary indicator belonging
# to the same multiple-response question.
############################################################

def _resolved_values(series, val_labels):
    """Return unique non-null label-resolved values for a series."""
    non_null = series.dropna().astype(str)
    if val_labels:
        non_null = non_null.map(lambda v: val_labels.get(v, v))
    return non_null.unique()


def _looks_binary(series, val_labels):
    """Return True if the column has only 1-4 distinct non-null values (binary/dichotomy)."""
    vals = _resolved_values(series, val_labels)
    return 0 < len(vals) <= 4


def build_multi_select_groups(df, meta, col_label_map, value_label_map):
    """
    Three-pass detection of multi-select column groups for SAV files.

    Pass 1 – mr_sets metadata (authoritative SPSS MR definitions).
    Pass 2 – columns sharing the same column label (same question, different options).
    Pass 3 – columns whose resolved values are exclusively Checked / Unchecked,
              grouped by shared label.

    Returns
    -------
    groups : dict  { group_label : [col_name, ...] }
        Each value list has >= 2 members and represents one multi-select question.
    col_to_group : dict  { col_name : group_label }
        Reverse lookup so the main loop can check membership quickly.
    """

    groups = {}  # label -> [cols]
    col_to_group = {}  # col  -> label

    # ----------------------------------------------------------
    # PASS 1: meta.mr_sets
    # ----------------------------------------------------------
    if meta is not None and hasattr(meta, "mr_sets") and meta.mr_sets:
        for set_name, mr_info in meta.mr_sets.items():
            var_list = mr_info.get("variable_list", [])
            if len(var_list) < 2:
                continue
            # Use the MR-set label if present, else the set name
            group_label = (mr_info.get("label") or set_name).strip()
            # Resolve to actual df column names
            resolved = [v for v in var_list if v in df.columns]
            if len(resolved) < 2:
                continue
            key = group_label
            suffix = 0
            while key in groups:
                suffix += 1
                key = f"{group_label}_{suffix}"
            groups[key] = resolved
            for col in resolved:
                col_to_group[col] = key

        if groups:
            print(f"  [MR detection] Pass 1 (mr_sets): {len(groups)} group(s) found")
            return groups, col_to_group

    # ----------------------------------------------------------
    # PASS 2: same column label  (>= 2 columns share identical label)
    # ----------------------------------------------------------
    label_to_cols = {}
    for col, label in col_label_map.items():
        if col not in df.columns:
            continue
        label_to_cols.setdefault(label, []).append(col)

    for label, cols in label_to_cols.items():
        if len(cols) < 2:
            continue
        # Every member must look like a binary/dichotomous variable
        # (<=4 unique non-null values) to avoid grouping genuinely
        # different questions that happen to share a label.
        if all(_looks_binary(df[c], value_label_map.get(c, {})) for c in cols):
            key = label
            suffix = 0
            while key in groups:
                suffix += 1
                key = f"{label}_{suffix}"
            groups[key] = cols
            for col in cols:
                col_to_group[col] = key

    # ----------------------------------------------------------
    # PASS 3: Checked / Unchecked columns not yet claimed
    # ----------------------------------------------------------
    CHECKED_LABELS = {"checked", "unchecked"}
    for col in df.columns:
        if col in col_to_group:
            continue
        resolved_values = _resolved_values(df[col], value_label_map.get(col, {}))
        unique_lower = {str(v).strip().lower() for v in resolved_values if pd.notna(v)}
        if unique_lower and unique_lower.issubset(CHECKED_LABELS):
            label = col_label_map.get(col, col)
            # Find any existing group that has a member with the same label
            matched_key = None
            for k, v_list in groups.items():
                if col_label_map.get(v_list[0], v_list[0]) == label:
                    matched_key = k
                    break
            if matched_key:
                groups[matched_key].append(col)
                col_to_group[col] = matched_key
            else:
                # Start a new tentative group (may be pruned below)
                groups[label] = groups.get(label, []) + [col]
                col_to_group[col] = label

    # Prune single-member groups (can't be multi-select alone)
    to_remove = [k for k, v in groups.items() if len(v) < 2]
    for k in to_remove:
        for col in groups[k]:
            del col_to_group[col]
        del groups[k]

    if groups:
        print(f"  [MR detection] Pass 2+3 (label inference): {len(groups)} group(s) found")
    else:
        print(f"  [MR detection] No multi-select groups detected")

    return groups, col_to_group


# Build the groups dict once, used throughout the rest of the pipeline
if input_path.endswith(".sav"):
    multi_select_groups, col_to_mr_group = build_multi_select_groups(
        df, meta, col_label_map, value_label_map
    )
else:
    # CSV path still uses pipe-delimiter detection; no pre-built groups
    multi_select_groups = {}
    col_to_mr_group = {}

############################################################
# BUILD LOOKUP
############################################################

lookup_rows = []

q_counter = 1
a_counter = 1

# Track which MR groups have already been emitted into the lookup
# so we only create the parent row once per group.
emitted_mr_groups = {}  # group_label -> parent_code

for col in tqdm(df.columns, desc="Building lookup", unit="col"):

    if col == "respondent_id":
        continue

    values = df[col].dropna().astype(str)
    question_text_proc = col_label_map.get(col, col).strip()

    col_value_labels = value_label_map.get(col, {})
    if col_value_labels:
        values = values.map(lambda v: col_value_labels.get(v, v))

    ########################################################
    # DETERMINE IF THIS COLUMN BELONGS TO AN MR GROUP
    ########################################################

    mr_group_key = col_to_mr_group.get(col)

    ########################################################
    # BUILD PARENT QUESTION CODE
    ########################################################

    if mr_group_key is not None:
        # All columns in the same MR group share one parent code
        if mr_group_key in emitted_mr_groups:
            parent_code = emitted_mr_groups[mr_group_key]
        else:
            # Assign a new code for this group
            group_label = mr_group_key  # group key == label
            if "?" in group_label:
                parent_code = f"Q{q_counter}"
                q_counter += 1
            else:
                parent_code = f"A{a_counter}"
                a_counter += 1
            emitted_mr_groups[mr_group_key] = parent_code
    else:
        if "?" in question_text_proc:
            parent_code = f"Q{q_counter}"
            q_counter += 1
        else:
            parent_code = f"A{a_counter}"
            a_counter += 1

    ########################################################
    # DETECT TYPE
    ########################################################

    if mr_group_key is not None:
        qtype = "multi_select"
    elif values.str.contains(r"\|").any():
        qtype = "multi_select"
    elif is_free_text(col, df[col]):
        qtype = "free_text"
    else:
        qtype = "single_choice"

    ########################################################
    # MULTISELECT  — SAV dichotomy style (one col per option)
    ########################################################

    if qtype == "multi_select" and mr_group_key is not None:

        # The option label is the column label itself (stripped of the parent question)
        # For dichotomy sets the variable label IS the option name.
        # For category sets the option name comes from value labels — but since
        # each variable in a category MR set stores an integer code, we use
        # the column label as the option label (SPSS convention).
        option_text = question_text_proc

        # Use the MR group's label as the canonical question text
        group_question_text = mr_group_key

        # Sub-column code  e.g. Q3_R0, Q3_R1 …
        # Count how many sub-cols this group already has
        n_existing = sum(
            1 for r in lookup_rows
            if r.get("parent_question_code") == parent_code and r.get("response_code") == 1
        )
        question_code = f"{parent_code}_R{n_existing + 1}"

        lookup_rows.append({
            "question_text": f"{question_code} {group_question_text}",
            "question_text_proc": group_question_text,
            "question_code": question_code,
            "parent_question_code": parent_code,
            "question_type": qtype,
            "response_code": 0,
            "response_text": "Unchecked",
            "natural_language_map": group_question_text,
            "response_text_proc": "Unchecked",
            "source_varname": col,
        })

        lookup_rows.append({
            "question_text": f"{question_code} {group_question_text}",
            "question_text_proc": group_question_text,
            "question_code": question_code,
            "parent_question_code": parent_code,
            "question_type": qtype,
            "response_code": 1,
            "response_text": option_text,
            "natural_language_map": group_question_text,
            "response_text_proc": option_text,
            "source_varname": col,
        })

    ########################################################
    # MULTISELECT  — CSV pipe-delimited style
    ########################################################

    elif qtype == "multi_select":

        options = (
            values
            .str.split("|")
            .explode()
            .str.strip()
            .dropna()
            .unique()
        )

        for i, opt in enumerate(sorted(options), start=0):
            question_code = f"{parent_code}_R{i+1}"

            lookup_rows.append({
                "question_text": f"{question_code} {question_text_proc}",
                "question_text_proc": question_text_proc,
                "question_code": question_code,
                "parent_question_code": parent_code,
                "question_type": qtype,
                "response_code": 0,
                "response_text": "Unchecked",
                "natural_language_map": question_text_proc,
                "response_text_proc": "Unchecked",
                "source_varname": col,
            })

            lookup_rows.append({
                "question_text": f"{question_code} {question_text_proc}",
                "question_text_proc": question_text_proc,
                "question_code": question_code,
                "parent_question_code": parent_code,
                "question_type": qtype,
                "response_code": 1,
                "response_text": opt,
                "natural_language_map": question_text_proc,
                "response_text_proc": opt,
                "source_varname": col,
            })

    ########################################################
    # FREE TEXT — one lookup row per unique value, single encoded column
    ########################################################

    elif qtype == "free_text":

        unique_vals = values.unique()

        if len(unique_vals) == 0:
            lookup_rows.append({
                "question_text": f"{parent_code} {question_text_proc}",
                "question_text_proc": question_text_proc,
                "question_code": parent_code,
                "parent_question_code": parent_code,
                "question_type": qtype,
                "response_code": 0,
                "response_text": "",
                "natural_language_map": question_text_proc,
                "response_text_proc": "",
                "source_varname": col,
            })
        else:
            for i, opt in enumerate(sorted(unique_vals), start=0):
                lookup_rows.append({
                    "question_text": f"{parent_code} {question_text_proc}",
                    "question_text_proc": question_text_proc,
                    "question_code": parent_code,
                    "parent_question_code": parent_code,
                    "question_type": qtype,
                    "response_code": i,
                    "response_text": opt,
                    "natural_language_map": question_text_proc,
                    "response_text_proc": opt,
                    "source_varname": col,
                })

    ########################################################
    # SINGLE CHOICE
    ########################################################

    else:

        options = values.unique()

        if len(options) == 0:
            lookup_rows.append({
                "question_text": f"{parent_code} {question_text_proc}",
                "question_text_proc": question_text_proc,
                "question_code": parent_code,
                "parent_question_code": parent_code,
                "question_type": qtype,
                "response_code": 0,
                "response_text": "",
                "natural_language_map": question_text_proc,
                "response_text_proc": "",
                "source_varname": col,
            })
        else:
            for i, opt in enumerate(sorted(options), start=0):
                lookup_rows.append({
                    "question_text": f"{parent_code} {question_text_proc}",
                    "question_text_proc": question_text_proc,
                    "question_code": parent_code,
                    "parent_question_code": parent_code,
                    "question_type": qtype,
                    "response_code": i,
                    "response_text": opt,
                    "natural_language_map": question_text_proc,
                    "response_text_proc": opt,
                    "source_varname": col,
                })

lookup = pd.DataFrame(lookup_rows)

############################################################
# BUILD ENCODED MATRIX
############################################################

encoded_cols = {}

############################################################
# MULTISELECT EXPANSION — SAV dichotomy style
#
# Each source column IS the indicator; "Checked" (or the
# counted_value) maps to response_code=1, everything else=0.
############################################################

CHECKED_STRINGS = {"checked", "2", "yes", "true", "1"}  # broad net; refine if needed

mr_multi_questions = lookup.loc[
    (lookup.question_type == "multi_select") &
    (lookup.source_varname.isin(col_to_mr_group.keys())),
    "question_code"
].unique()

for qcode in tqdm(mr_multi_questions, desc="Encoding MR multi-select", unit="q"):
    row = lookup.loc[lookup.question_code == qcode].iloc[0]
    source_col = row["source_varname"]

    col_val_labels = value_label_map.get(source_col, {})

    raw = df[source_col].astype(str).where(~df[source_col].isna(), other=np.nan)
    resolved = raw.map(lambda v: col_val_labels.get(str(v), str(v)) if pd.notna(v) else v)

    # Determine what counts as "checked" for this column
    # Check if the mr_set specifies a counted_value (for category sets)
    counted_value = None
    if meta is not None and hasattr(meta, "mr_sets") and meta.mr_sets:
        group_key = col_to_mr_group.get(source_col)
        if group_key:
            # Find the matching mr_set entry
            for set_name, mr_info in meta.mr_sets.items():
                lbl = (mr_info.get("label") or set_name).strip()
                if lbl == group_key and mr_info.get("counted_value") is not None:
                    counted_value = str(mr_info["counted_value"])
                    break

    if counted_value is not None:
        # Category MR set: "checked" = the counted_value in the raw numeric column
        encoded_cols[qcode] = df[source_col].apply(
            lambda v: 1 if pd.notna(v) and str(v) == counted_value else (0 if pd.notna(v) else np.nan)
        )
    else:
        # Dichotomy MR set or label-inferred: "checked" = resolved label in CHECKED_STRINGS
        encoded_cols[qcode] = resolved.apply(
            lambda v: 1 if pd.notna(v) and str(v).strip().lower() in CHECKED_STRINGS else (0 if pd.notna(v) else np.nan)
        )

############################################################
# MULTISELECT EXPANSION — CSV pipe-delimited style
############################################################

pipe_multi_questions = lookup.loc[
    (lookup.question_type == "multi_select") &
    (~lookup.source_varname.isin(col_to_mr_group.keys())),
    "parent_question_code"
].unique()

for parent in tqdm(pipe_multi_questions, desc="Encoding pipe multi-select", unit="q"):

    source_varname = lookup.loc[
        lookup.parent_question_code == parent,
        "source_varname"
    ].iloc[0]

    parent_lookup = lookup.loc[
        (lookup.parent_question_code == parent) & (lookup.response_code == 1)
        ].sort_values("question_code")

    split_vals = df[source_varname].str.split("|")  # NaN stays NaN

    for _, lrow in parent_lookup.iterrows():
        qcode = lrow["question_code"]
        option_text = lrow["response_text_proc"]

        encoded_cols[qcode] = split_vals.apply(
            lambda x: (1 if option_text in [v.strip() for v in x] else 0) if isinstance(x, list) else np.nan
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
    "source_varname",  # new required column
]

missing_cols = [c for c in required_cols if c not in lookup.columns]

if missing_cols:
    raise ValueError(f"Lookup missing columns: {missing_cols}")

# Report how much of the SAV meta was consumed
if meta is not None:
    n_cols = len(df.columns)
    n_labelled = sum(1 for c in df.columns if c in value_label_map and value_label_map[c])
    n_measured = sum(1 for c in df.columns if variable_measure.get(c, "") in ("nominal", "ordinal", "scale"))
    n_col_labels = sum(1 for c in df.columns if col_label_map.get(c, c) != c)
    n_mr_groups = len(multi_select_groups)
    n_mr_cols = len(col_to_mr_group)
    print(f"\n=== SAV META COVERAGE ===")
    print(f"  Columns with human labels (column_names_to_labels): {n_col_labels}/{n_cols}")
    print(f"  Columns with value labels (variable_value_labels):  {n_labelled}/{n_cols}")
    print(f"  Columns with measure type (variable_measure):       {n_measured}/{n_cols}")
    print(f"  Multiple-response groups detected:                  {n_mr_groups}")
    print(f"  Columns assigned to MR groups:                      {n_mr_cols}/{n_cols}")

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
    if col in col_to_mr_group:
        # Each MR member column -> one encoded sub-column; already counted
        # Only count it once (the first time we encounter it)
        expected_encoded_cols += 1
    else:
        values = df[col].dropna().astype(str)
        if values.str.contains(r"\|").any():
            n_options = values.str.split("|").explode().str.strip().dropna().unique()
            expected_encoded_cols += len(n_options)
        else:
            expected_encoded_cols += 1  # single_choice, free_text = 1 column

if len(encoded.columns) == expected_encoded_cols:
    ok(f"Column count matches: {len(encoded.columns)}")
else:
    fail(f"Column count mismatch: expected={expected_encoded_cols}, encoded={len(encoded.columns)}")

# ----------------------------------------------------------
# CHECK 3: Missing values (-1 codes)
# ----------------------------------------------------------

expected_minus1 = 0
for col in df.columns:
    n_missing = original_missing[col].sum()
    if n_missing == 0:
        continue
    if col in col_to_mr_group:
        expected_minus1 += n_missing  # one encoded col per MR source col
    elif df[col].dropna().astype(str).str.contains(r"\|").any():
        n_subcols = lookup.loc[
            (lookup.source_varname == col) & (lookup.response_code == 1),
            "question_code"
        ].nunique()
        expected_minus1 += n_missing * n_subcols
    else:
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
    encoded_row = encoded.loc[rid]
    errors = []

    for col in df.columns:
        orig_val = original_row[col]

        col_val_labels = value_label_map.get(col, {})
        orig_val_str = col_val_labels.get(str(orig_val), str(orig_val)) if not pd.isna(orig_val) else orig_val

        if pd.isna(orig_val):
            q_codes = lookup.loc[lookup.source_varname == col, "question_code"].unique()
            for qc in q_codes:
                if qc in encoded_row.index and encoded_row[qc] != -1:
                    errors.append(f"{col}: expected -1 for missing, got {encoded_row[qc]}")
            continue

        col_label = col_label_map.get(col, col)

        # MR group member — check that "checked" state round-trips
        if col in col_to_mr_group:
            qc_rows = lookup.loc[
                (lookup.source_varname == col) & (lookup.response_code == 1),
                "question_code"
            ]
            if qc_rows.empty:
                continue
            qc = qc_rows.iloc[0]
            if qc in encoded_row.index:
                enc_val = encoded_row[qc]
                resolved_str = str(orig_val_str).strip().lower()
                expect_checked = (
                        resolved_str in CHECKED_STRINGS or
                        (meta is not None and hasattr(meta, "mr_sets") and any(
                            str(mr_info.get("counted_value", "")) == str(orig_val)
                            for mr_info in meta.mr_sets.values()
                            if col in mr_info.get("variable_list", [])
                        ))
                )
                expected_enc = 1 if expect_checked else 0
                if enc_val != expected_enc:
                    errors.append(
                        f"{col} (MR): orig='{orig_val_str}' expected enc={expected_enc}, got {enc_val}"
                    )
            continue

        values = df[col].dropna().astype(str)
        is_multi = values.str.contains(r"\|").any()

        if is_multi:
            checked_options = []
            sub = lookup.loc[
                (lookup.source_varname == col) & (lookup.response_code == 1)
                ]
            for _, lrow in sub.iterrows():
                qc = lrow["question_code"]
                if qc in encoded_row.index and encoded_row[qc] == 1:
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
n_questions = len(encoded.columns)
total_cells = n_respondents * n_questions

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
# 3. Duplicate respondents
# ----------------------------------------------------------

print(f"\n  Duplicate detection")

n_dupes = encoded.duplicated().sum()
pct_dupes = 100 * n_dupes / n_respondents
print(f"  Exact duplicate rows: {n_dupes} / {n_respondents}  ({pct_dupes:.1f}%)")
if n_dupes > 0:
    print(f"  ⚠️  WARNING: {n_dupes} respondents have identical response patterns")

# ----------------------------------------------------------
# 4. Lookup table integrity
# ----------------------------------------------------------

print(f"\n  Lookup table")
n_lookup_cols = lookup["source_varname"].nunique()
n_encoded_cols = len(encoded.columns)
n_orphan_enc = sum(1 for c in encoded.columns if c not in lookup["question_code"].values)
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
# 5. Question type breakdown
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

lookup['parent_question_code'] = lookup['parent_question_code'].str.lower()

encoded.to_csv(os.path.join(output_dir, "encoded_response_matrix.csv"))
lookup.drop(columns=["source_varname"]).to_csv(os.path.join(output_dir, "lookup_table.csv"), index=False)

print("\nPipeline completed")
print("Encoded shape:", encoded.shape)
print("Lookup rows:", len(lookup))

print(f'Output at: {output_dir}')
print(f'response_matrix at: {os.path.join(output_dir, "encoded_response_matrix.csv")}')
print(f'lookup_table: {os.path.join(output_dir, "lookup_table.csv")}')