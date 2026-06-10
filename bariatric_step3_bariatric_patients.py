"""
bariatric_step3_bariatric_patients.py

Streams procedure.csv and diagnosis.csv from GCS.
Filters to diabetic gastroparesis patients from Step 2.

Filters applied:
    1. CPT in {43775, 43644, 43645, 43846, 43847}
    2. Age >= 18 at time of surgery (year_of_birth from patient.csv)
    3. Surgery date >= 2016-01-01
    4. Gastroparesis diagnosis before surgery
    5. First qualifying bariatric surgery only (post-GP dx)
    6. Flag patients with prior bariatric surgery before GP dx
    7. Flag same-date combined procedures (coding error / unusual)
    8. Diabetes diagnosis before surgery date

Requires:
    gp_dm_patients.csv (from Step 2)

Outputs:
    bariatric_study_patients.csv
        (patient_id, first_gp_date, first_dm_date,
         bariatric_date, cpt_code, procedure_type,
         prior_bariatric_flag, combined_procedure_flag)
"""

import pandas as pd
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROC_FILE  = f"{BUCKET}/procedure.csv"
DIAG_FILE  = f"{BUCKET}/diagnosis.csv"
PAT_FILE   = f"{BUCKET}/patient.csv"
CHUNK_SIZE = 100_000

SLEEVE_CODES  = {"43775"}
BYPASS_CODES  = {"43644", "43645", "43846", "43847"}
RARE_CODES    = {"43645", "43847"}
BARIATRIC_ALL = SLEEVE_CODES | BYPASS_CODES

MIN_SURGERY_DATE = pd.Timestamp("2016-01-01")
MIN_AGE          = 18

DIABETES_PREFIX = {"E08", "E09", "E10", "E11", "E12", "E13"}

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
# LOAD STEP 2 OUTPUT
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3: FINDING BARIATRIC SURGERY PATIENTS")
print("=" * 60)

gp_dm = pd.read_csv("gp_dm_patients.csv", dtype=str)
gp_dm["first_gp_date"] = parse_dates(gp_dm["first_gp_date"])
gp_dm_ids = set(gp_dm["patient_id"].unique())
print(f"  GP + diabetes patients from Step 2: {len(gp_dm_ids):,}")

# ─────────────────────────────────────────────────────────────
# LOAD PATIENT FILE — get year_of_birth for age filter
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

# Filter to our cohort only
pat_df = pat_df[pat_df["patient_id"].isin(gp_dm_ids)][["patient_id", "year_of_birth"]]
yob_dict = dict(zip(pat_df["patient_id"], pat_df["year_of_birth"]))
print(f"  Year of birth loaded for {len(yob_dict):,} patients")

# ─────────────────────────────────────────────────────────────
# PASS 1: STREAM PROCEDURE FILE
# ─────────────────────────────────────────────────────────────

print("\n  Streaming procedure.csv from GCS in chunks...")

proc_file = subprocess.Popen(
    ["gsutil", "cat", PROC_FILE],
    stdout=subprocess.PIPE
)
if proc_file.stdout is None:
    raise RuntimeError("Failed to open GCS stream for procedure.csv")

# Store all bariatric procedure rows per patient
# patient_id -> list of (date, cpt_code)
all_bar_procs = defaultdict(list)
total_rows = 0
chunk_num  = 0

for chunk in pd.read_csv(proc_file.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()

    required = {"patient_id", "code", "date"}
    if not required.issubset(chunk.columns):
        raise ValueError(f"Missing columns in procedure chunk: {list(chunk.columns)}")

    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    # Filter to our cohort and bariatric codes only
    chunk = chunk[
        chunk["patient_id"].isin(gp_dm_ids) &
        chunk["code"].isin(BARIATRIC_ALL)
    ].copy()

    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    for row in chunk.itertuples(index=False):
        all_bar_procs[row.patient_id].append((row.date, row.code))

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} procedure rows...")

proc_file.stdout.close()
ret = proc_file.wait()
if ret != 0:
    raise RuntimeError(f"gsutil exited with code {ret} — procedure.csv stream may be incomplete")

print(f"\n  Total procedure rows processed: {total_rows:,}")
print(f"  Patients with any bariatric CPT: {len(all_bar_procs):,}")

# ─────────────────────────────────────────────────────────────
# APPLY FILTERS PER PATIENT
# ─────────────────────────────────────────────────────────────

print("\n  Applying filters...")

# Build dict for fast GP date lookup instead of repeated dataframe scanning
gp_date_dict = dict(zip(gp_dm["patient_id"], gp_dm["first_gp_date"]))

results      = []
n_no_bar     = 0   # no bariatric after GP dx
n_too_early  = 0   # surgery before 2016
n_too_young  = 0   # age < 18
n_no_post_gp = 0   # no surgery after GP dx

for pid, procs in all_bar_procs.items():
    first_gp_date = gp_date_dict[pid]
    yob           = yob_dict.get(pid)

    # Sort procedures by date
    procs_sorted = sorted(procs, key=lambda x: x[0])

    # Filter 6: flag prior bariatric surgery (before GP dx)
    prior_bariatric = any(d < first_gp_date for d, _ in procs_sorted)

    # Filter 4+5: keep only surgeries AFTER GP dx
    post_gp_procs = [(d, c) for d, c in procs_sorted if d > first_gp_date]

    if len(post_gp_procs) == 0:
        n_no_post_gp += 1
        continue

    # Filter 3: surgery >= 2016
    post_gp_procs = [(d, c) for d, c in post_gp_procs if d >= MIN_SURGERY_DATE]

    if len(post_gp_procs) == 0:
        n_too_early += 1
        continue

    # Filter 5: first qualifying surgery
    index_date, index_cpt = post_gp_procs[0]

    # Filter 2: age >= 18 at surgery
    if yob is not None and not pd.isna(yob):
        age_at_surgery = index_date.year - int(yob)
        if age_at_surgery < MIN_AGE:
            n_too_young += 1
            continue
    else:
        age_at_surgery = None   # missing YOB — keep but flag

    # Filter 7: flag same-date combined procedures
    same_date_codes = {c for d, c in procs_sorted if d == index_date}
    combined_flag   = len(same_date_codes) > 1

    # Procedure type label
    proc_type = "sleeve" if index_cpt in SLEEVE_CODES else "bypass"
    rare_flag = index_cpt in RARE_CODES

    results.append({
        "patient_id":             pid,
        "first_gp_date":          first_gp_date,
        "bariatric_date":         index_date,
        "cpt_code":               index_cpt,
        "procedure_type":         proc_type,
        "rare_cpt_flag":          rare_flag,
        "prior_bariatric_flag":   prior_bariatric,
        "combined_procedure_flag": combined_flag,
        "age_at_surgery":         age_at_surgery,
    })

print(f"\n  Excluded — no surgery after GP dx:   {n_no_post_gp:,}")
print(f"  Excluded — surgery before 2016:      {n_too_early:,}")
print(f"  Excluded — age < 18 at surgery:      {n_too_young:,}")

# ─────────────────────────────────────────────────────────────
# PASS 2: STREAM DIAGNOSIS FILE — get first diabetes date
# Filter 8: diabetes diagnosis before surgery
# ─────────────────────────────────────────────────────────────

print("\n  Pass 2: getting first diabetes date per patient...")
print("  Streaming diagnosis.csv from GCS in chunks...")

study_ids    = {r["patient_id"] for r in results}
first_dm_date = {}   # patient_id -> earliest diabetes date

proc_diag = subprocess.Popen(
    ["gsutil", "cat", DIAG_FILE],
    stdout=subprocess.PIPE
)
if proc_diag.stdout is None:
    raise RuntimeError("Failed to open GCS stream for diagnosis.csv")

total_rows2 = 0
chunk_num2  = 0

for chunk in pd.read_csv(proc_diag.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows2 += len(chunk)
    chunk_num2  += 1

    chunk = chunk[chunk["patient_id"].isin(study_ids)].copy()

    if chunk.empty:
        continue

    chunk["prefix"] = chunk["code"].str[:3]
    dm_chunk = chunk[chunk["prefix"].isin(DIABETES_PREFIX)].copy()

    if len(dm_chunk) == 0:
        continue

    dm_chunk["date"] = parse_dates(dm_chunk["date"])
    dm_chunk = dm_chunk.dropna(subset=["date"])

    for row in dm_chunk.itertuples(index=False):
        pid = row.patient_id
        if pid not in first_dm_date or row.date < first_dm_date[pid]:
            first_dm_date[pid] = row.date

    if chunk_num2 % 50 == 0:
        print(f"  Processed {total_rows2:,} diagnosis rows...")

proc_diag.stdout.close()
ret2 = proc_diag.wait()
if ret2 != 0:
    raise RuntimeError(f"gsutil exited with code {ret2} — diagnosis.csv stream may be incomplete")

print(f"\n  Total diagnosis rows processed: {total_rows2:,}")

# ─────────────────────────────────────────────────────────────
# APPLY FILTER 8: diabetes before surgery
# ─────────────────────────────────────────────────────────────

final_results = []
n_no_dm_before = 0

for r in results:
    pid          = r["patient_id"]
    dm_date      = first_dm_date.get(pid)
    bar_date     = r["bariatric_date"]

    if dm_date is None or dm_date >= bar_date:
        n_no_dm_before += 1
        continue

    r["first_dm_date"] = dm_date
    final_results.append(r)

print(f"  Excluded — no diabetes before surgery: {n_no_dm_before:,}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 SUMMARY")
print("=" * 60)
print(f"  GP + diabetes patients entering Step 3:  {len(gp_dm_ids):,}")
print(f"  With any bariatric CPT (any timing):     {len(all_bar_procs):,}")
print(f"  After all filters:                       {len(final_results):,}")

# CPT breakdown
df = pd.DataFrame(final_results)
if len(df) > 0:
    print("\n  Procedure type breakdown:")
    for cpt in sorted(BARIATRIC_ALL):
        n     = (df["cpt_code"] == cpt).sum()
        label = "sleeve" if cpt in SLEEVE_CODES else "bypass"
        flag  = " *** RARE ***" if cpt in RARE_CODES else ""
        print(f"    {cpt} ({label}): {n:,}{flag}")

    print(f"\n  Sleeve patients:                   {(df['procedure_type'] == 'sleeve').sum():,}")
    print(f"  Bypass patients:                   {(df['procedure_type'] == 'bypass').sum():,}")
    print(f"  Prior bariatric surgery flagged:   {df['prior_bariatric_flag'].sum():,}")
    print(f"  Combined procedure flagged:        {df['combined_procedure_flag'].sum():,}")
    print(f"  Missing age (YOB not available):   {df['age_at_surgery'].isna().sum():,}")

    if len(df) < 100:
        print("\n  WARNING: Fewer than 100 patients — discuss new TriNetX pull with PI.")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────────────────────

    col_order = [
        "patient_id", "first_gp_date", "first_dm_date",
        "bariatric_date", "cpt_code", "procedure_type",
        "rare_cpt_flag", "prior_bariatric_flag",
        "combined_procedure_flag", "age_at_surgery"
    ]
    df[col_order].to_csv("bariatric_study_patients.csv", index=False)
    print("\n  Saved: bariatric_study_patients.csv")
    print("  Run bariatric_step4_a1c.py next.")
else:
    print("\n  ERROR: No patients passed all filters.")
    print("  Check CPT codes — may not be present in this dataset.")
