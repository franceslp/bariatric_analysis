"""
bariatric_task1_time_to_surgery.py

Calculates time from gastroparesis diagnosis to bariatric surgery
for the 1,153 patient cohort from Step 3.

Requires:
    bariatric_study_patients.csv (from Step 3)

Outputs:
    time_to_surgery_summary.txt
    time_to_surgery_distribution.csv
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("TASK 1: TIME FROM GP DIAGNOSIS TO BARIATRIC SURGERY")
print("=" * 60)

df = pd.read_csv("bariatric_study_patients.csv", dtype=str)

# Safety check for required columns
required_cols = {"first_gp_date", "bariatric_date", "procedure_type"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in bariatric_study_patients.csv: {missing}")

df["first_gp_date"]  = pd.to_datetime(df["first_gp_date"],  errors="coerce")
df["bariatric_date"] = pd.to_datetime(df["bariatric_date"], errors="coerce")
df = df.dropna(subset=["first_gp_date", "bariatric_date"])
print(f"  Patients loaded: {len(df):,}")

# ─────────────────────────────────────────────────────────────
# CALCULATE TIME TO SURGERY
# ─────────────────────────────────────────────────────────────

df["days_to_surgery"]   = (df["bariatric_date"] - df["first_gp_date"]).dt.days
df["months_to_surgery"] = df["days_to_surgery"] / 30.44
df["years_to_surgery"]  = df["days_to_surgery"] / 365.25

# FIX 4: negative values indicate a serious Step 3 error — raise, don't warn
negative_n = (df["days_to_surgery"] < 0).sum()
if negative_n > 0:
    raise ValueError(
        f"{negative_n} patients have negative time-to-surgery — "
        "check Step 3 GP/surgery date filtering"
    )
print("  Data quality check passed: no negative time-to-surgery values")

# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────

days   = df["days_to_surgery"]
months = df["months_to_surgery"]

print("\n  Time from GP diagnosis to bariatric surgery:")
print(f"    Mean:    {months.mean():.1f} months ({months.mean()/12:.1f} years)")
print(f"    Median:  {months.median():.1f} months ({months.median()/12:.1f} years)")
print(f"    Std dev: {months.std():.1f} months")
print(f"    Min:     {months.min():.1f} months")
print(f"    Max:     {months.max():.1f} months ({months.max()/12:.1f} years)")
print(f"    IQR:     {months.quantile(0.25):.1f} – {months.quantile(0.75):.1f} months")

# FIX 5: within 1yr/2yr — likely goes directly in results section
within_1yr = (months < 12).sum()
within_2yr = (months < 24).sum()
print(f"\n  Within 1 year of GP diagnosis:  {within_1yr:,} ({within_1yr/len(df)*100:.1f}%)")
print(f"  Within 2 years of GP diagnosis: {within_2yr:,} ({within_2yr/len(df)*100:.1f}%)")

# Percentiles
print("\n  Percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"    {p}th: {months.quantile(p/100):.1f} months")

# ─────────────────────────────────────────────────────────────
# BREAKDOWN BY PROCEDURE TYPE
# ─────────────────────────────────────────────────────────────

print("\n  By procedure type:")
for ptype in ["sleeve", "bypass"]:
    sub = df[df["procedure_type"] == ptype]["months_to_surgery"]

    # FIX 1: handle empty groups gracefully
    if len(sub) == 0:
        print(f"\n    {ptype.capitalize()}: No patients")
        continue

    print(f"\n    {ptype.capitalize()} (n={len(sub):,}):")
    print(f"      Mean:   {sub.mean():.1f} months ({sub.mean()/12:.1f} years)")
    print(f"      Median: {sub.median():.1f} months ({sub.median()/12:.1f} years)")
    print(f"      IQR:    {sub.quantile(0.25):.1f} – {sub.quantile(0.75):.1f} months")

# ─────────────────────────────────────────────────────────────
# PRIOR BARIATRIC FLAG
# ─────────────────────────────────────────────────────────────

if "prior_bariatric_flag" in df.columns:
    df["prior_bariatric_flag"] = df["prior_bariatric_flag"].astype(str).str.lower().isin(["true", "1"])
    n_prior = df["prior_bariatric_flag"].sum()
    print(f"\n  Prior bariatric surgery flagged: {n_prior:,} ({n_prior/len(df)*100:.1f}%)")
    print("  Consider sensitivity analysis excluding these patients.")

# ─────────────────────────────────────────────────────────────
# DISTRIBUTION BY TIME BUCKET
# ─────────────────────────────────────────────────────────────

bins   = [0, 6, 12, 24, 36, 60, float("inf")]
labels = ["<6mo", "6-12mo", "1-2yr", "2-3yr", "3-5yr", ">5yr"]

df["time_bucket"] = pd.cut(
    df["months_to_surgery"],
    bins=bins,
    labels=labels,
    right=False
)

print("\n  Distribution by time to surgery:")
bucket_counts = df["time_bucket"].value_counts().reindex(labels, fill_value=0)
for label, count in bucket_counts.items():
    pct = count / len(df) * 100
    print(f"    {label:>8}: {count:>5,}  ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────
# BREAKDOWN BY YEAR OF SURGERY
# ─────────────────────────────────────────────────────────────

print("\n  By year of surgery:")
df["surgery_year"] = df["bariatric_date"].dt.year
year_stats = df.groupby("surgery_year")["months_to_surgery"].agg(
    n="count",
    median="median"
).reset_index()

for _, row in year_stats.iterrows():
    print(f"    {int(row['surgery_year'])}: n={int(row['n']):,}  median {row['median']:.1f} months to surgery")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

# FIX 3: preserve flags in distribution file for sensitivity analyses
dist_cols = ["patient_id", "first_gp_date", "bariatric_date",
             "procedure_type", "days_to_surgery",
             "months_to_surgery", "time_bucket"]

for flag_col in ["prior_bariatric_flag", "combined_procedure_flag"]:
    if flag_col in df.columns:
        dist_cols.insert(4, flag_col)

df[dist_cols].to_csv("time_to_surgery_distribution.csv", index=False)

# Pre-compute sleeve/bypass series for safe summary generation
sleeve = df[df["procedure_type"] == "sleeve"]["months_to_surgery"]
bypass = df[df["procedure_type"] == "bypass"]["months_to_surgery"]
sleeve_median_str = f"{sleeve.median():.1f} months ({sleeve.median()/12:.1f} years)" if len(sleeve) else "N/A"
bypass_median_str = f"{bypass.median():.1f} months ({bypass.median()/12:.1f} years)" if len(bypass) else "N/A"

# FIX 2: include years in summary file
summary_lines = [
    "TIME FROM GP DIAGNOSIS TO BARIATRIC SURGERY",
    "=" * 50,
    f"N = {len(df):,}",
    f"Mean:   {months.mean():.1f} months ({months.mean()/12:.1f} years)",
    f"Median: {months.median():.1f} months ({months.median()/12:.1f} years)",
    f"IQR:    {months.quantile(0.25):.1f} – {months.quantile(0.75):.1f} months",
    f"Min:    {months.min():.1f} months",
    f"Max:    {months.max():.1f} months ({months.max()/12:.1f} years)",
    "",
    "PERCENTILES:",
] + [f"  {p}th: {months.quantile(p/100):.1f} months" for p in [10, 25, 50, 75, 90]] + [
    "",
    f"Within 1 year:  {within_1yr:,} ({within_1yr/len(df)*100:.1f}%)",
    f"Within 2 years: {within_2yr:,} ({within_2yr/len(df)*100:.1f}%)",
    "",
    "SLEEVE:",
    f"  n={len(df[df['procedure_type']=='sleeve']):,}",
    f"  Median: {sleeve_median_str}",
    "",
    "BYPASS:",
    f"  n={len(bypass):,}",
    f"  Median: {bypass_median_str}",
]

with open("time_to_surgery_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n  Saved: time_to_surgery_summary.txt")
print("  Saved: time_to_surgery_distribution.csv")
