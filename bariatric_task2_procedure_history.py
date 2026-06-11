"""
bariatric_task2_procedure_history.py

Streams procedure.csv from GCS.
For each of the 1,153 study patients, finds all procedures
performed BETWEEN their gastroparesis diagnosis and bariatric surgery.

Window: first_gp_date <= procedure_date < bariatric_date
This captures the GP disease management period before surgery,
which is more clinically meaningful than lifetime procedure history.

Requires:
    bariatric_study_patients.csv (from Step 3)

Outputs:
    procedure_history_counts.csv
    procedure_history_summary.txt
"""

import pandas as pd
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
PROC_FILE  = f"{BUCKET}/procedure.csv"
CHUNK_SIZE = 100_000

# Exclude bariatric surgery codes themselves from history
BARIATRIC_CODES = {"43775", "43644", "43645", "43846", "43847"}

# ─────────────────────────────────────────────────────────────
# LOAD STUDY COHORT
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("TASK 2: PROCEDURE HISTORY (GP DIAGNOSIS → SURGERY)")
print("Window: first_gp_date <= procedure_date < bariatric_date")
print("=" * 60)

df = pd.read_csv("bariatric_study_patients.csv", dtype=str)
df["bariatric_date"] = pd.to_datetime(df["bariatric_date"], errors="coerce")
df["first_gp_date"]  = pd.to_datetime(df["first_gp_date"],  errors="coerce")
n_before = len(df)
df = df.dropna(subset=["bariatric_date", "first_gp_date"])
n_after = len(df)
if n_before != n_after:
    print(f"  WARNING: dropped {n_before - n_after:,} patients with missing GP or surgery dates")

# Fast lookups
surgery_date_dict = dict(zip(df["patient_id"], df["bariatric_date"]))
gp_date_dict      = dict(zip(df["patient_id"], df["first_gp_date"]))
study_ids         = set(df["patient_id"].unique())
n_patients        = len(study_ids)

print(f"  Study patients loaded: {n_patients:,}")

# ─────────────────────────────────────────────────────────────
# DATE PARSER
# ─────────────────────────────────────────────────────────────

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
# STREAM PROCEDURE FILE
# ─────────────────────────────────────────────────────────────

print("\n  Streaming procedure.csv from GCS in chunks...")

proc = subprocess.Popen(
    ["gsutil", "cat", PROC_FILE],
    stdout=subprocess.PIPE
)
if proc.stdout is None:
    raise RuntimeError("Failed to open GCS stream for procedure.csv")

# cpt_code -> set of patient_ids who had it in the window
cpt_patient_counter     = defaultdict(set)
patients_with_any_proc  = set()
total_rows              = 0
chunk_num               = 0

for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
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

    # Filter to study patients only
    chunk = chunk[chunk["patient_id"].isin(study_ids)].copy()

    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    # Map GP date and surgery date for window filtering
    chunk["gp_date"]      = chunk["patient_id"].map(gp_date_dict)
    chunk["surgery_date"] = chunk["patient_id"].map(surgery_date_dict)

    # Restrict to GP diagnosis → surgery window
    chunk = chunk[
        (chunk["date"] >= chunk["gp_date"]) &
        (chunk["date"] <  chunk["surgery_date"])
    ]

    if chunk.empty:
        continue

    # Exclude blank codes and bariatric codes themselves
    chunk = chunk[chunk["code"] != ""]
    chunk = chunk[~chunk["code"].isin(BARIATRIC_CODES)]

    if chunk.empty:
        continue

    # Count unique patients per CPT code
    for pid, code in zip(chunk["patient_id"], chunk["code"]):
        cpt_patient_counter[code].add(pid)
        patients_with_any_proc.add(pid)

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} rows...")

proc.stdout.close()
ret = proc.wait()
if ret != 0:
    raise RuntimeError(f"gsutil exited with code {ret}")

print(f"\n  Total procedure rows processed: {total_rows:,}")
print(f"  Unique CPT codes found in window: {len(cpt_patient_counter):,}")
n_with_proc = len(patients_with_any_proc)
print(f"  Patients with ≥1 procedure in window: {n_with_proc:,} ({n_with_proc/n_patients*100:.1f}%)")
print(f"  Patients with no procedures in window: {n_patients - n_with_proc:,} ({(n_patients - n_with_proc)/n_patients*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# BUILD RESULTS TABLE
# ─────────────────────────────────────────────────────────────

results = []
for code, patient_set in cpt_patient_counter.items():
    n = len(patient_set)
    results.append({
        "cpt_code":       code,
        "n_patients":     n,
        "pct_of_cohort":  round(n / n_patients * 100, 1)
    })

if len(results) == 0:
    raise ValueError(
        "No procedures found between GP diagnosis and surgery — "
        "check date window and patient IDs"
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("n_patients", ascending=False).reset_index(drop=True)
results_df["rank"] = range(1, len(results_df) + 1)
results_df = results_df[["rank", "cpt_code", "n_patients", "pct_of_cohort"]]

# ─────────────────────────────────────────────────────────────
# PRINT TOP 50
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TOP 50 PROCEDURES DURING GP DISEASE COURSE PRE-SURGERY")
print(f"(% of {n_patients:,} study patients who had each procedure)")
print("=" * 60)
print(f"  {'CPT Code':<12} {'N Patients':>12} {'% of Cohort':>12}")
print(f"  {'-'*12} {'-'*12} {'-'*12}")

for _, row in results_df.head(50).iterrows():
    print(f"  {row['cpt_code']:<12} {int(row['n_patients']):>12,} {row['pct_of_cohort']:>11.1f}%")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

results_df.to_csv("procedure_history_counts.csv", index=False)

with open("procedure_history_summary.txt", "w") as f:
    f.write("PROCEDURE HISTORY: GP DIAGNOSIS TO BARIATRIC SURGERY\n")
    f.write("Window: first_gp_date <= procedure_date < bariatric_date\n")
    f.write("=" * 50 + "\n")
    f.write(f"Study patients: {n_patients:,}\n")
    f.write(f"Patients with ≥1 procedure in window: {n_with_proc:,} ({n_with_proc/n_patients*100:.1f}%)\n")
    f.write(f"Unique CPT codes found: {len(cpt_patient_counter):,}\n\n")
    f.write("TOP 50 CPT CODES (by unique patient count):\n")
    f.write(f"{'CPT Code':<12} {'N Patients':>12} {'% Cohort':>10}\n")
    f.write("-" * 36 + "\n")
    for _, row in results_df.head(50).iterrows():
        f.write(f"{row['cpt_code']:<12} {int(row['n_patients']):>12,} {row['pct_of_cohort']:>9.1f}%\n")

print("\n  Saved: procedure_history_counts.csv")
print("  Saved: procedure_history_summary.txt")
print("\n  NOTE: CPT codes are not labeled here.")
print("  Cross-reference with a CPT lookup table to identify")
print("  procedure names before presenting to PI.")
