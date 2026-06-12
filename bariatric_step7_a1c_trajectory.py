"""
bariatric_step7_a1c_trajectory.py

Collects post-operative A1c values for matched cohorts.
Follows Sadda et al. (JAMA Surgery 2026) follow-up structure.

Follow-up structure (per Sadda):
    - Index date = bariatric surgery date
    - Follow-up begins 1 month post-op (day 31)
    - Annual windows:
        Year 1 = days 31-365
        Year 2 = days 366-730
        Year 3 = days 731-1095
        Year 4 = days 1096-1460
        Year 5 = days 1461-1825
    - At each window: most recent A1c value used
    - Physiologic filter: 3-20%
    - Baseline A1c: carried from Step 5 matched files (not re-collected)

Requires:
    matched_study.csv
    matched_comparison.csv
    matched_study_sensitivity.csv
    matched_comparison_sensitivity.csv

Outputs:
    a1c_trajectory_primary.csv
    a1c_trajectory_sensitivity.csv
    a1c_summary_primary.csv
    a1c_summary_sensitivity.csv
    a1c_available_counts_primary.csv
    a1c_available_counts_sensitivity.csv
"""

import pandas as pd
import numpy as np
import subprocess
import datetime

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET     = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
LAB_FILE   = f"{BUCKET}/lab_result.csv"
CHUNK_SIZE = 100_000

A1C_LOINC = {"4548-4", "17856-6", "4549-2"}

# Post-operative follow-up windows only (baseline from Step 5)
FOLLOW_UP_WINDOWS = {
    "year_1": (31,   365),
    "year_2": (366,  730),
    "year_3": (731,  1095),
    "year_4": (1096, 1460),
    "year_5": (1461, 1825),
}

A1C_MIN = 3.0
A1C_MAX = 20.0

print("=" * 60)
print("STEP 7: POST-OPERATIVE A1c TRAJECTORY COLLECTION")
print(f"  Run date: {datetime.date.today()}")
print(f"  Sadda ref: JAMA Surgery 2026, doi:10.1001/jamasurg.2026.1593")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# HELPERS
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

def assign_timepoint(days_from_surgery):
    for tp, (start, end) in FOLLOW_UP_WINDOWS.items():
        if start <= days_from_surgery <= end:
            return tp
    return None

# ─────────────────────────────────────────────────────────────
# LOAD MATCHED COHORTS
# ─────────────────────────────────────────────────────────────

print("\n  Loading matched cohorts...")

def load_cohort(study_file, comp_file, label):
    study = pd.read_csv(study_file, dtype=str)
    comp  = pd.read_csv(comp_file, dtype=str)
    study["bariatric_date"] = parse_dates(study["bariatric_date"])
    comp["bariatric_date"]  = parse_dates(comp["bariatric_date"])
    study["baseline_a1c"] = pd.to_numeric(study["baseline_a1c"], errors="coerce")
    comp["baseline_a1c"]  = pd.to_numeric(comp["baseline_a1c"], errors="coerce")
    assert study["baseline_a1c"].notna().all(),         f"Missing baseline_a1c in matched study cohort ({label}) — check Step 6 complete-case filter"
    assert comp["baseline_a1c"].notna().all(),         f"Missing baseline_a1c in matched comparison cohort ({label}) — check Step 6 complete-case filter"
    study["group"] = "study"
    comp["group"]  = "comparison"
    combined = pd.concat([study, comp], ignore_index=True)
    print(f"  {label}: {len(study):,} study + {len(comp):,} comparison = {len(combined):,} total")
    return combined

primary     = load_cohort("matched_study.csv", "matched_comparison.csv", "Primary")
sensitivity = load_cohort("matched_study_sensitivity.csv",
                          "matched_comparison_sensitivity.csv", "Sensitivity")

# All unique patients across both cohorts for single streaming pass
all_patients = pd.concat([primary, sensitivity]).drop_duplicates(subset=["patient_id"])
surgery_dict = dict(zip(all_patients["patient_id"], all_patients["bariatric_date"]))
all_ids      = set(all_patients["patient_id"].unique())

# Fix 1: assert bariatric_date complete in both cohorts
assert primary["bariatric_date"].notna().all(),     "NaT bariatric_date in primary cohort — check Step 6 output"
assert sensitivity["bariatric_date"].notna().all(),     "NaT bariatric_date in sensitivity cohort — check Step 6 output"

# Fix 4: assert no patient has multiple surgery dates across cohorts
dup_check = (
    pd.concat([primary, sensitivity])
    .groupby("patient_id")["bariatric_date"]
    .nunique()
)
assert (dup_check <= 1).all(),     "Patient has multiple bariatric dates across primary/sensitivity cohorts"

print(f"\n  Total unique patients: {len(all_ids):,}")
print(f"  Baseline A1c already available from Step 5 matched files")
print(f"  Collecting post-op A1c for years 1-5 only...")

# ─────────────────────────────────────────────────────────────
# STREAM LAB FILE
# ─────────────────────────────────────────────────────────────

print("\n  Streaming lab_result.csv from GCS...")
print("  (This may take substantial time depending on dataset size)")

proc = subprocess.Popen(
    ["gsutil", "cat", LAB_FILE],
    stdout=subprocess.PIPE
)
if proc.stdout is None:
    raise RuntimeError("Failed to open GCS stream for lab_result.csv")

# {patient_id: {timepoint: (date, value)}}
a1c_data = {}

total_rows     = 0
chunk_num      = 0
a1c_rows_found = 0

for chunk in pd.read_csv(proc.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    chunk = chunk[
        chunk["patient_id"].isin(all_ids) &
        chunk["code"].isin(A1C_LOINC)
    ].copy()

    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    chunk["lab_result_num_val"] = pd.to_numeric(
        chunk["lab_result_num_val"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["lab_result_num_val"])
    chunk = chunk[
        (chunk["lab_result_num_val"] >= A1C_MIN) &
        (chunk["lab_result_num_val"] <= A1C_MAX)
    ]

    if chunk.empty:
        continue

    a1c_rows_found += len(chunk)

    for row in chunk.itertuples(index=False):
        pid     = row.patient_id
        surg_dt = surgery_dict.get(pid)
        if surg_dt is None:
            continue

        days = (row.date - surg_dt).days
        tp   = assign_timepoint(days)
        if tp is None:
            continue

        # Keep most recent A1c within each timepoint window (per Sadda)
        if pid not in a1c_data:
            a1c_data[pid] = {}
        if tp not in a1c_data[pid] or row.date > a1c_data[pid][tp][0]:
            a1c_data[pid][tp] = (row.date, row.lab_result_num_val)

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} lab rows... ({a1c_rows_found:,} A1c values found)")

proc.stdout.close()
ret = proc.wait()
if ret != 0:
    raise RuntimeError(f"gsutil exited with code {ret} — lab_result.csv may be incomplete")

print(f"\n  Total lab rows processed:  {total_rows:,}")
print(f"  Total A1c values found:    {a1c_rows_found:,}")
print(f"  Patients with any post-op A1c: {len(a1c_data):,}")

# ─────────────────────────────────────────────────────────────
# BUILD TRAJECTORY DATAFRAME
# ─────────────────────────────────────────────────────────────

print("\n  Building trajectory dataframes...")

TIMEPOINTS = ["baseline"] + list(FOLLOW_UP_WINDOWS.keys())

def build_trajectory(cohort_df, label):
    rows = []

    for _, pt in cohort_df.iterrows():
        pid      = pt["patient_id"]
        group    = pt["group"]
        surg_dt  = pt["bariatric_date"]
        pt_a1c   = a1c_data.get(pid, {})

        # Baseline: use Step 5 value (internally consistent)
        baseline_val = pt["baseline_a1c"]
        if pd.notna(baseline_val):
            rows.append({
                "patient_id":        pid,
                "group":             group,
                "timepoint":         "baseline",
                "a1c_date":          pd.NaT,
                "a1c_value":         baseline_val,
                "days_from_surgery": None
            })

        # Post-op timepoints: from lab streaming
        for tp in FOLLOW_UP_WINDOWS.keys():
            if tp in pt_a1c:
                date, value = pt_a1c[tp]
                rows.append({
                    "patient_id":        pid,
                    "group":             group,
                    "timepoint":         tp,
                    "a1c_date":          date,
                    "a1c_value":         value,
                    "days_from_surgery": (date - surg_dt).days
                })

    df = pd.DataFrame(rows)
    print(f"\n  {label}: {len(df):,} patient-timepoint observations")
    return df

primary_traj     = build_trajectory(primary,     "Primary")
sensitivity_traj = build_trajectory(sensitivity, "Sensitivity")

# ─────────────────────────────────────────────────────────────
# BUILD SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def build_summary(traj_df, label):
    rows = []
    for tp in TIMEPOINTS:
        tp_df = traj_df[traj_df["timepoint"] == tp]
        for group in ["study", "comparison"]:
            g_vals = tp_df[tp_df["group"] == group]["a1c_value"]
            if len(g_vals) == 0:
                continue
            rows.append({
                "timepoint":  tp,
                "group":      group,
                "n":          len(g_vals),
                "mean_a1c":   round(g_vals.mean(), 3),
                "sd_a1c":     round(g_vals.std(), 3),
                "median_a1c": round(g_vals.median(), 3),
                "q1_a1c":     round(g_vals.quantile(0.25), 3),
                "q3_a1c":     round(g_vals.quantile(0.75), 3),
            })

    summary = pd.DataFrame(rows)

    print(f"\n  {label} A1c SUMMARY:")
    print(f"  {'Timepoint':<12} {'Group':<12} {'N':>6} {'Mean A1c':>10} {'SD':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*6} {'-'*10} {'-'*8}")
    for _, row in summary.iterrows():
        print(f"  {row['timepoint']:<12} {row['group']:<12} "
              f"{int(row['n']):>6,} {row['mean_a1c']:>10.2f}% {row['sd_a1c']:>8.2f}")

    return summary

primary_summary     = build_summary(primary_traj,     "PRIMARY")
sensitivity_summary = build_summary(sensitivity_traj, "SENSITIVITY")

# ─────────────────────────────────────────────────────────────
# A1c AVAILABLE COUNTS TABLE
# Note: this counts patients WITH an A1c value at each timepoint
# NOT patients censored/at-risk (which requires formal survival analysis)
# ─────────────────────────────────────────────────────────────

def build_available_counts(traj_df, label):
    rows = []
    for tp in TIMEPOINTS:
        tp_df   = traj_df[traj_df["timepoint"] == tp]
        n_study = (tp_df["group"] == "study").sum()
        n_comp  = (tp_df["group"] == "comparison").sum()
        rows.append({
            "timepoint":    tp,
            "study_n":      n_study,
            "comparison_n": n_comp
        })

    counts_df = pd.DataFrame(rows)

    print(f"\n  {label} A1c AVAILABLE AT TIMEPOINT:")
    print(f"  {'Timepoint':<12} {'Study':>8} {'Comparison':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*12}")
    for _, row in counts_df.iterrows():
        print(f"  {row['timepoint']:<12} {int(row['study_n']):>8,} {int(row['comparison_n']):>12,}")

    return counts_df

primary_counts     = build_available_counts(primary_traj,     "PRIMARY")
sensitivity_counts = build_available_counts(sensitivity_traj, "SENSITIVITY")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

primary_traj.to_csv("a1c_trajectory_primary.csv", index=False)
sensitivity_traj.to_csv("a1c_trajectory_sensitivity.csv", index=False)
primary_summary.to_csv("a1c_summary_primary.csv", index=False)
sensitivity_summary.to_csv("a1c_summary_sensitivity.csv", index=False)
primary_counts.to_csv("a1c_available_counts_primary.csv", index=False)
sensitivity_counts.to_csv("a1c_available_counts_sensitivity.csv", index=False)

print("\n  Saved: a1c_trajectory_primary.csv")
print("  Saved: a1c_trajectory_sensitivity.csv")
print("  Saved: a1c_summary_primary.csv")
print("  Saved: a1c_summary_sensitivity.csv")
print("  Saved: a1c_available_counts_primary.csv")
print("  Saved: a1c_available_counts_sensitivity.csv")
print("\n  Next: run bariatric_step8_outcomes.py")
