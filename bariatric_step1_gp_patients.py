"""
bariatric_step1_gp_patients.py

Streams diagnosis.csv from GCS in chunks.
Finds all patients with K31.84 (gastroparesis).

Outputs:
    gp_patients.csv (patient_id, first_gp_date)
"""

import pandas as pd
import subprocess

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
DIAG_FILE  = f"{BUCKET}/diagnosis.csv"
CHUNK_SIZE = 100_000

GASTROPARESIS_CODE = "K31.84"

# ─────────────────────────────────────────────────────────────
# DATE PARSER
# ─────────────────────────────────────────────────────────────

def parse_dates(series):
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask],
            format="mixed",
            dayfirst=False,
            errors="coerce"
        )
    return result

# ─────────────────────────────────────────────────────────────
# STREAM FILE
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: FINDING GASTROPARESIS PATIENTS (K31.84)")
print("Streaming diagnosis.csv from GCS in chunks...")
print("=" * 60)

# NOTE: no text=True — pandas reads binary streams more reliably
proc = subprocess.Popen(
    ["gsutil", "cat", DIAG_FILE],
    stdout=subprocess.PIPE
)

gp_dict    = {}   # patient_id -> earliest GP date
total_rows = 0
chunk_num  = 0

for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue
    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["code"] = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    gp_chunk = chunk.loc[
        chunk["code"] == GASTROPARESIS_CODE,
        ["patient_id", "date"]
    ]

    if len(gp_chunk) == 0:
        continue

    gp_chunk = gp_chunk.copy()
    gp_chunk["date"] = parse_dates(gp_chunk["date"])
    gp_chunk = gp_chunk.dropna(subset=["date"])

    # Track earliest GP date per patient
    for pid, dt in zip(gp_chunk["patient_id"], gp_chunk["date"]):
        if pid not in gp_dict or dt < gp_dict[pid]:
            gp_dict[pid] = dt

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} rows...")

# ─────────────────────────────────────────────────────────────
# FINALIZE
# ─────────────────────────────────────────────────────────────

print(f"\n  Total diagnosis rows processed: {total_rows:,}")

if len(gp_dict) == 0:
    raise ValueError("No K31.84 gastroparesis cases found in dataset.")

gp_patients = pd.DataFrame(
    [(pid, dt) for pid, dt in gp_dict.items()],
    columns=["patient_id", "first_gp_date"]
)

print(f"  Gastroparesis patients found: {len(gp_patients):,}")

gp_patients.to_csv("gp_patients.csv", index=False)
print("\n  Saved: gp_patients.csv")
print("  Run bariatric_step2_dm_patients.py next.")
