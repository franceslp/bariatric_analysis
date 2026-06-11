"""
bariatric_step4_comparison_group.py

Builds the comparison group for the bariatric glycemic outcomes study.
Diabetic bariatric patients WITHOUT prior gastroparesis.

DESIGN NOTE:
The comparison group is defined as bariatric patients who did NOT have
gastroparesis diagnosed BEFORE surgery. Patients who develop gastroparesis
AFTER surgery are retained and flagged (post_op_gp_flag=True) rather than
excluded. This allows measurement of incident post-operative gastroparesis
and avoids throwing away a clinically meaningful signal.

SELECTION BIAS NOTE:
This comparison group excludes patients with gastroparesis before surgery,
meaning it is healthier by construction. GP may be a marker of autonomic
dysfunction and disease severity. This should be acknowledged in the
methods section as a potential source of upward bias in the control group's
glycemic outcomes.

Filters applied:
    1. CPT in {43775, 43644, 43645, 43846, 43847}
    2. Age >= 18 at time of surgery
    3. Surgery date >= 2016-01-01
    4. NO K31.84 gastroparesis BEFORE surgery date
    5. Diabetes diagnosis (E08-E13) before surgery
    6. First bariatric surgery only
    7. Flag patients who develop gastroparesis AFTER surgery
    8. Flag same-date combined procedures

Requires:
    (no prior step outputs — reads directly from GCS)

Outputs:
    comparison_group_patients.csv
"""

import pandas as pd
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET    = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROC_FILE = f"{BUCKET}/procedure.csv"
DIAG_FILE = f"{BUCKET}/diagnosis.csv"
PAT_FILE  = f"{BUCKET}/patient.csv"
CHUNK_SIZE = 100_000

SLEEVE_CODES  = {"43775"}
BYPASS_CODES  = {"43644", "43645", "43846", "43847"}
RARE_CODES    = {"43645", "43847"}
BARIATRIC_ALL = SLEEVE_CODES | BYPASS_CODES

GASTROPARESIS_CODE = "K31.84"
DIABETES_PREFIX    = {"E08", "E09", "E10", "E11", "E12", "E13"}

MIN_SURGERY_DATE = pd.Timestamp("2016-01-01")
MIN_AGE          = 18

def parse_dates(series):
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    return result

# ─────────────────────────────────────────────────────────────
# PASS 1: STREAM DIAGNOSIS FILE
# Collect: (a) earliest GP date per patient (for pre/post-op classification)
#          (b) first diabetes date per patient
# NOTE: we do NOT build an exclusion set here — GP timing relative to
# surgery determines inclusion, not GP presence alone (Fix 1 / Option B)
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4: BUILDING COMPARISON GROUP")
print("(Diabetic bariatric patients without pre-op gastroparesis)")
print("=" * 60)

print("\n  Pass 1: Streaming diagnosis.csv — collecting GP and DM dates...")

proc_diag = subprocess.Popen(
    ["gsutil", "cat", DIAG_FILE],
    stdout=subprocess.PIPE
)
if proc_diag.stdout is None:
    raise RuntimeError("Failed to open GCS stream for diagnosis.csv")

first_gp_date = {}    # patient_id -> earliest GP date (any time)
first_dm_date = {}    # patient_id -> earliest diabetes date
total_rows    = 0
chunk_num     = 0

for chunk in pd.read_csv(proc_diag.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()

    required = {"patient_id", "code", "date"}
    if not required.issubset(chunk.columns):
        raise ValueError(f"Missing columns: {list(chunk.columns)}")

    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    # Collect earliest GP date per patient
    gp_chunk = chunk[chunk["code"] == GASTROPARESIS_CODE].copy()
    if len(gp_chunk) > 0:
        gp_chunk["date"] = parse_dates(gp_chunk["date"])
        gp_chunk = gp_chunk.dropna(subset=["date"])
        for pid, dt in zip(gp_chunk["patient_id"], gp_chunk["date"]):
            if pid not in first_gp_date or dt < first_gp_date[pid]:
                first_gp_date[pid] = dt

    # Collect earliest diabetes date per patient
    chunk["prefix"] = chunk["code"].str[:3]
    dm_chunk = chunk[chunk["prefix"].isin(DIABETES_PREFIX)].copy()
    if len(dm_chunk) > 0:
        dm_chunk["date"] = parse_dates(dm_chunk["date"])
        dm_chunk = dm_chunk.dropna(subset=["date"])
        for pid, dt in zip(dm_chunk["patient_id"], dm_chunk["date"]):
            if pid not in first_dm_date or dt < first_dm_date[pid]:
                first_dm_date[pid] = dt

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} diagnosis rows...")

proc_diag.stdout.close()
ret = proc_diag.wait()
if ret != 0:
    raise RuntimeError(f"gsutil exited with code {ret} — diagnosis.csv stream incomplete")

print(f"\n  Total diagnosis rows processed: {total_rows:,}")
print(f"  Patients with any GP (K31.84):     {len(first_gp_date):,}")
print(f"  Patients with any diabetes (E08-E13): {len(first_dm_date):,}")

# ─────────────────────────────────────────────────────────────
# LOAD PATIENT FILE — year_of_birth
# ─────────────────────────────────────────────────────────────

print("\n  Loading patient.csv for year_of_birth...")

proc_pat = subprocess.Popen(
    ["gsutil", "cat", PAT_FILE],
    stdout=subprocess.PIPE
)
if proc_pat.stdout is None:
    raise RuntimeError("Failed to open GCS stream for patient.csv")

pat_df = pd.read_csv(proc_pat.stdout, dtype=str, low_memory=False)
proc_pat.stdout.close()
proc_pat.wait()

pat_df.columns = pat_df.columns.str.strip().str.lower()
pat_df["patient_id"]    = pat_df["patient_id"].fillna("").astype(str)
pat_df["year_of_birth"] = pd.to_numeric(pat_df["year_of_birth"], errors="coerce")
yob_dict = dict(zip(pat_df["patient_id"], pat_df["year_of_birth"]))
print(f"  Year of birth loaded for {len(yob_dict):,} patients")

# ─────────────────────────────────────────────────────────────
# PASS 2: STREAM PROCEDURE FILE
# Collect all bariatric procedures
# No upfront exclusion — GP timing applied per patient below
# ─────────────────────────────────────────────────────────────

print("\n  Pass 2: Streaming procedure.csv — collecting bariatric procedures...")

proc_file = subprocess.Popen(
    ["gsutil", "cat", PROC_FILE],
    stdout=subprocess.PIPE
)
if proc_file.stdout is None:
    raise RuntimeError("Failed to open GCS stream for procedure.csv")

all_bar_procs = defaultdict(list)
total_rows2   = 0
chunk_num2    = 0

for chunk in pd.read_csv(proc_file.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()

    required = {"patient_id", "code", "date"}
    if not required.issubset(chunk.columns):
        raise ValueError(f"Missing columns: {list(chunk.columns)}")

    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows2 += len(chunk)
    chunk_num2  += 1

    chunk = chunk[chunk["code"].isin(BARIATRIC_ALL)].copy()

    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    for row in chunk.itertuples(index=False):
        all_bar_procs[row.patient_id].append((row.date, row.code))

    if chunk_num2 % 50 == 0:
        print(f"  Processed {total_rows2:,} procedure rows...")

proc_file.stdout.close()
ret2 = proc_file.wait()
if ret2 != 0:
    raise RuntimeError(f"gsutil exited with code {ret2} — procedure.csv stream incomplete")

print(f"\n  Total procedure rows processed: {total_rows2:,}")
print(f"  Patients with any bariatric CPT: {len(all_bar_procs):,}")

# ─────────────────────────────────────────────────────────────
# APPLY FILTERS PER PATIENT
# ─────────────────────────────────────────────────────────────

print("\n  Applying filters...")

results           = []
n_gp_before_surg  = 0    # excluded: GP diagnosed before surgery
n_no_dm           = 0
n_too_early       = 0
n_too_young       = 0
n_dm_after_surg   = 0

for pid, procs in all_bar_procs.items():
    yob     = yob_dict.get(pid)
    dm_date = first_dm_date.get(pid)
    gp_date = first_gp_date.get(pid)

    # Must have diabetes
    if dm_date is None:
        n_no_dm += 1
        continue

    # Sort procedures by date
    procs_sorted = sorted(procs, key=lambda x: x[0])

    # Filter 3: surgery >= 2016
    post_2016_procs = [(d, c) for d, c in procs_sorted if d >= MIN_SURGERY_DATE]
    if len(post_2016_procs) == 0:
        n_too_early += 1
        continue

    # Filter 6: first bariatric surgery only
    index_date, index_cpt = post_2016_procs[0]

    # Filter 2: age >= 18 at surgery
    if yob is not None and not pd.isna(yob):
        age_at_surgery = index_date.year - int(yob)
        if age_at_surgery < MIN_AGE:
            n_too_young += 1
            continue
    else:
        age_at_surgery = None

    # Filter 5: diabetes before surgery
    if dm_date >= index_date:
        n_dm_after_surg += 1
        continue

    # FIX 1 / Option B: exclude only if GP diagnosed BEFORE surgery
    # Patients with GP after surgery are retained and flagged
    if gp_date is not None and gp_date < index_date:
        n_gp_before_surg += 1
        continue

    # Flag 7: post-operative gastroparesis (GP diagnosed after surgery)
    post_op_gp_flag = (
        gp_date is not None and gp_date >= index_date
    )

    # Flag 8: same-date combined procedures
    same_date_codes = {c for d, c in procs_sorted if d == index_date}
    combined_flag   = len(same_date_codes) > 1

    proc_type = "sleeve" if index_cpt in SLEEVE_CODES else "bypass"
    rare_flag = index_cpt in RARE_CODES

    results.append({
        "patient_id":              pid,
        "first_dm_date":           dm_date,
        "bariatric_date":          index_date,
        "cpt_code":                index_cpt,
        "procedure_type":          proc_type,
        "rare_cpt_flag":           rare_flag,
        "combined_procedure_flag": combined_flag,
        "post_op_gp_flag":         post_op_gp_flag,
        "age_at_surgery":          age_at_surgery,
    })

print(f"\n  Excluded — GP before surgery:        {n_gp_before_surg:,}")
print(f"  Excluded — no diabetes:              {n_no_dm:,}")
print(f"  Excluded — surgery before 2016:      {n_too_early:,}")
print(f"  Excluded — age < 18:                 {n_too_young:,}")
print(f"  Excluded — diabetes after surgery:   {n_dm_after_surg:,}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 SUMMARY")
print("=" * 60)

df = pd.DataFrame(results)

if len(df) == 0:
    raise ValueError("No patients passed filters — check data and criteria")

n_post_op_gp = df["post_op_gp_flag"].sum()

print(f"  Patients with any bariatric CPT:     {len(all_bar_procs):,}")
print(f"  After all filters (comparison group):{len(df):,}")
print(f"  Post-op gastroparesis flagged:       {n_post_op_gp:,} ({n_post_op_gp/len(df)*100:.1f}%)")

print("\n  Procedure type breakdown:")
for cpt in sorted(BARIATRIC_ALL):
    n     = (df["cpt_code"] == cpt).sum()
    label = "sleeve" if cpt in SLEEVE_CODES else "bypass"
    flag  = " *** RARE ***" if cpt in RARE_CODES else ""
    print(f"    {cpt} ({label}): {n:,}{flag}")

sleeve_n = (df["procedure_type"] == "sleeve").sum()
bypass_n = (df["procedure_type"] == "bypass").sum()
print(f"\n  Sleeve total:                  {sleeve_n:,}")
print(f"  Bypass total:                  {bypass_n:,}")
print(f"  Combined procedure flagged:    {df['combined_procedure_flag'].sum():,}")
print(f"  Missing age (YOB not avail):   {df['age_at_surgery'].isna().sum():,}")
print(f"\n  Study group (Step 3):          1,153")
print(f"  Comparison group (Step 4):     {len(df):,}")
if len(df) >= 1153:
    print(f"  Ratio:                         1 : {len(df)//1153}")

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────

col_order = [
    "patient_id", "first_dm_date", "bariatric_date",
    "cpt_code", "procedure_type", "rare_cpt_flag",
    "combined_procedure_flag", "post_op_gp_flag",
    "age_at_surgery"
]
df[col_order].to_csv("comparison_group_patients.csv", index=False)
print("\n  Saved: comparison_group_patients.csv")
print("  Next: run bariatric_step5_baseline_a1c.py")
