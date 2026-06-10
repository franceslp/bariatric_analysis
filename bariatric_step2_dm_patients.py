"""
bariatric_step2_dm_patients.py

Streams diagnosis.csv from GCS in chunks.
Filters to gastroparesis patients from Step 1.
Finds which of those also have diabetes (E08-E13).

Definition: "lifetime ICD-based diabetes comorbidity"
Any E08-E13 code at any point in the patient record.
Temporal relationship to gastroparesis is not assessed here.

Requires:
    gp_patients.csv (from Step 1)

Outputs:
    gp_dm_patients.csv (patient_id, first_gp_date)
"""

import pandas as pd
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAG_FILE  = f"{BUCKET}/diagnosis.csv"
CHUNK_SIZE = 100_000

DIABETES_PREFIX = {"E08", "E09", "E10", "E11", "E12", "E13"}

# ─────────────────────────────────────────────────────────────
# LOAD STEP 1 OUTPUT
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2: FINDING DIABETIC GASTROPARESIS PATIENTS")
print("Definition: lifetime ICD-based diabetes comorbidity")
print("(any E08-E13 code ever in record — not time-bounded)")
print("=" * 60)

gp_patients = pd.read_csv("gp_patients.csv", dtype=str)
gp_ids = set(gp_patients["patient_id"].unique())
print(f"  GP patients loaded from Step 1: {len(gp_ids):,}")

# ─────────────────────────────────────────────────────────────
# STREAM DIAGNOSIS FILE — single pass
# ─────────────────────────────────────────────────────────────

print("\n  Streaming diagnosis.csv from GCS in chunks...")

proc = subprocess.Popen(
    ["gsutil", "cat", DIAG_FILE],
    stdout=subprocess.PIPE
)
if proc.stdout is None:
    raise RuntimeError("Failed to open GCS stream for diagnosis.csv")

# FIX 4: defaultdict(set) — cleaner and faster than manual key checks
patient_prefix = defaultdict(set)
total_rows     = 0
chunk_num      = 0

for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()

    # FIX 2: validate required columns exist
    required = {"patient_id", "code"}
    if not required.issubset(chunk.columns):
        raise ValueError(f"Missing columns in chunk: {list(chunk.columns)}")

    # FIX 1: clean patient_id before filtering
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    chunk = chunk[chunk["patient_id"].isin(gp_ids)].copy()

    if chunk.empty:
        continue

    chunk["prefix"] = chunk["code"].str[:3]
    dm_chunk = chunk[chunk["prefix"].isin(DIABETES_PREFIX)].copy()

    if len(dm_chunk) == 0:
        continue

    for pid, prefix in zip(dm_chunk["patient_id"], dm_chunk["prefix"]):
        patient_prefix[pid].add(prefix)

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} rows...")

# FIX 3: check gsutil exit code — catch silent truncation
proc.stdout.close()
ret = proc.wait()
if ret != 0:
    raise RuntimeError(
        f"gsutil exited with code {ret} — diagnosis.csv stream may be incomplete"
    )

# ─────────────────────────────────────────────────────────────
# FINALIZE
# ─────────────────────────────────────────────────────────────

print(f"\n  Total diagnosis rows processed: {total_rows:,}")

dm_ids = set(patient_prefix.keys())
print(f"  GP patients with lifetime diabetes (E08-E13): {len(dm_ids):,}")

if len(dm_ids) == 0:
    raise ValueError("No diabetic gastroparesis patients found.")

print("\n  Breakdown by diabetes type (unique patients, not mutually exclusive):")
print("  Note: conditional prevalence — P(diabetes type | gastroparesis)")
from collections import Counter
counter = Counter()
for prefixes in patient_prefix.values():
    for p in prefixes:
        counter[p] += 1
for prefix in sorted(DIABETES_PREFIX):
    print(f"    {prefix}.x: {counter[prefix]:,}")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────────────────────

gp_dm_patients = gp_patients[gp_patients["patient_id"].isin(dm_ids)].copy()
print(f"\n  Final GP + diabetes cohort: {len(gp_dm_patients):,}")

gp_dm_patients.to_csv("gp_dm_patients.csv", index=False)
print("\n  Saved: gp_dm_patients.csv")
print("  Run bariatric_step3_bariatric_patients.py next.")
