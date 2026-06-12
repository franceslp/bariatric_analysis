"""
bariatric_step6_propensity_matching.py

Propensity score matching for bariatric glycemic outcomes study.
Follows Sadda et al. (JAMA Surgery 2026) methodology.

Method:
    - Logistic regression to estimate propensity scores
    - 1:1 nearest-neighbor matching WITHOUT replacement
    - Primary caliper:     0.2 SD of logit(PS) (Rosenbaum & Rubin standard)
    - Sensitivity caliper: 0.1 SD of logit(PS) (stricter)
    - Balance criterion:   SMD < 0.1 for all covariates (per Sadda)
    - Complete case analysis: patients missing A1c or BMI excluded

Requires:
    study_covariates.csv      (from Step 5)
    comparison_covariates.csv (from Step 5)

Outputs:
    matched_study.csv
    matched_comparison.csv
    matched_study_sensitivity.csv
    matched_comparison_sensitivity.csv
    balance_table.csv         (SMD before/after matching — Sadda eTable 2 equivalent)
    ps_overlap_summary.txt    (positivity check)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PRIMARY_CALIPER     = 0.2   # Rosenbaum & Rubin standard
SENSITIVITY_CALIPER = 0.1   # Stricter sensitivity analysis
RANDOM_SEED         = 42

print("=" * 60)
print("STEP 6: PROPENSITY SCORE MATCHING")
print(f"  Run date:          {datetime.date.today()}")
print(f"  Primary caliper:   {PRIMARY_CALIPER} SD of logit(PS)")
print(f"  Sensitivity:       {SENSITIVITY_CALIPER} SD of logit(PS)")
print(f"  Matching:          1:1 nearest-neighbor, no replacement")
print(f"  Balance criterion: SMD < 0.1")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# COVARIATES FOR PS MODEL
# Continuous: standardized before fitting
# Binary: used as-is
# ─────────────────────────────────────────────────────────────

CONTINUOUS_VARS = [
    "age_at_surgery",
    "baseline_a1c",
    "baseline_bmi",
    "dm_duration_years",
]

BINARY_VARS = [
    "procedure_type_sleeve",   # derived from procedure_type below
    "t1dm",
    "t2dm",
    "dm_renal",
    "dm_neuro",
    "dm_circ",
    "dm_opthal",
    "dm_other",
    "dyslipidemia",
    "ckd",
    "stroke",
    "cad",
    "heart_failure",
    "hypertension",
    "metformin",
    "any_insulin",
    "rapid_insulin",
    "long_insulin",
    "glp1",
    "sglt2",
    "dpp4",
    "sulfonylurea",
    "tzd",
    # Demographics
    "sex_female",              # derived from sex below
    "race_white",              # derived from race below
    "race_black",
    "ethnicity_hispanic",
    # Note: missingness indicators excluded from PS model
    # Excluded to maintain consistency with complete-case PS estimation framework
    # and avoid conditioning on post-selection artifacts (Sadda-style methodology)
]

ALL_COVARIATES = CONTINUOUS_VARS + BINARY_VARS

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

print("\n  Loading covariate files...")

study = pd.read_csv("study_covariates.csv", dtype=str)
comp  = pd.read_csv("comparison_covariates.csv", dtype=str)

print(f"  Study group loaded:      {len(study):,}")
print(f"  Comparison group loaded: {len(comp):,}")

# ─────────────────────────────────────────────────────────────
# TYPE CONVERSION
# ─────────────────────────────────────────────────────────────

def convert_types(df):
    # Continuous
    for col in CONTINUOUS_VARS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean columns stored as strings
    bool_cols = [c for c in df.columns if c not in
                 ["patient_id", "sex", "race", "ethnicity",
                  "procedure_type", "age_at_surgery"] + CONTINUOUS_VARS]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                {"True": 1, "False": 0, "1": 1, "0": 0,
                 True: 1, False: 0}
            ).fillna(0).astype(int)
    return df

study = convert_types(study)
comp  = convert_types(comp)

# ─────────────────────────────────────────────────────────────
# DERIVE BINARY INDICATOR VARIABLES
# ─────────────────────────────────────────────────────────────

for df in [study, comp]:
    df["procedure_type_sleeve"] = (df["procedure_type"] == "sleeve").astype(int)
    df["sex_female"]            = (df["sex"].str.lower() == "female").astype(int)
    df["race_white"]            = (df["race"].str.lower() == "white").astype(int)
    df["race_black"]            = (df["race"].str.lower().str.contains("black|african", na=False)).astype(int)
    df["ethnicity_hispanic"]    = (df["ethnicity"].str.lower().str.contains("hispanic|latino", na=False)).astype(int)

# ─────────────────────────────────────────────────────────────
# COMPLETE CASE ANALYSIS
# Drop patients missing A1c or BMI (per Sadda methodology)
# ─────────────────────────────────────────────────────────────

print("\n  Applying complete case filter (drop missing A1c or BMI)...")

n_study_before = len(study)
n_comp_before  = len(comp)

study = study.dropna(subset=["baseline_a1c", "baseline_bmi"]).copy()
comp  = comp.dropna(subset=["baseline_a1c", "baseline_bmi"]).copy()

print(f"  Study:      {n_study_before:,} → {len(study):,} "
      f"(dropped {n_study_before - len(study):,} missing A1c/BMI)")
print(f"  Comparison: {n_comp_before:,} → {len(comp):,} "
      f"(dropped {n_comp_before - len(comp):,} missing A1c/BMI)")

# Also drop missing dm_duration
study = study.dropna(subset=["dm_duration_years"]).copy()
comp  = comp.dropna(subset=["dm_duration_years"]).copy()
print(f"  After DM duration filter: study={len(study):,}, comp={len(comp):,}")

# ─────────────────────────────────────────────────────────────
# PREPARE MODEL MATRIX
# ─────────────────────────────────────────────────────────────

# Add cohort label
study["_group"] = 1   # study = gastroparesis bariatric
comp["_group"]  = 0   # comparison = non-GP bariatric

combined = pd.concat([study, comp], ignore_index=True)

# Fill any remaining NaN in binary vars with 0
for col in BINARY_VARS:
    if col in combined.columns:
        combined[col] = combined[col].fillna(0)

# Build feature matrix — only use columns that exist
available_covariates = [c for c in ALL_COVARIATES if c in combined.columns]
missing_covariates   = [c for c in ALL_COVARIATES if c not in combined.columns]
if missing_covariates:
    print(f"\n  WARNING: these covariates not found and will be skipped: {missing_covariates}")

X = combined[available_covariates].copy()
y = combined["_group"].values

print(f"\n  Covariates in PS model: {len(available_covariates)}")
print(f"  Total patients for matching: {len(combined):,} "
      f"(study={study['_group'].sum():,}, comp={(y==0).sum():,})")

# ─────────────────────────────────────────────────────────────
# STANDARDIZE CONTINUOUS VARIABLES
# NOTE: standardization applied here per Step 5 design note
# ─────────────────────────────────────────────────────────────

scaler = StandardScaler()
cont_cols_available = [c for c in CONTINUOUS_VARS if c in X.columns]
X[cont_cols_available] = scaler.fit_transform(X[cont_cols_available])

# ─────────────────────────────────────────────────────────────
# FIT PROPENSITY SCORE MODEL
# ─────────────────────────────────────────────────────────────

print("\n  Fitting logistic regression for propensity scores...")

# Fill any remaining NaN before fitting
# Explicit by variable type — avoids heuristic inference
for col in X.columns:
    if col in CONTINUOUS_VARS:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(0)
# Final safety check — no NaN allowed in model matrix
assert X.isna().sum().sum() == 0, "NaN values remain in model matrix after imputation"

ps_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    solver="lbfgs",
    C=1.0
)
ps_model.fit(X, y)

# Propensity scores
combined["ps"]       = ps_model.predict_proba(X)[:, 1]
combined["logit_ps"] = np.log(combined["ps"] / (1 - combined["ps"]))

print(f"  PS model converged: {ps_model.n_iter_[0]} iterations")
print(f"  Study PS:      mean={combined[combined['_group']==1]['ps'].mean():.3f}, "
      f"SD={combined[combined['_group']==1]['ps'].std():.3f}")
print(f"  Comparison PS: mean={combined[combined['_group']==0]['ps'].mean():.3f}, "
      f"SD={combined[combined['_group']==0]['ps'].std():.3f}")

# ─────────────────────────────────────────────────────────────
# POSITIVITY CHECK — PS overlap assessment
# ─────────────────────────────────────────────────────────────

print("\n  Positivity check (PS overlap)...")

study_ps = combined[combined["_group"] == 1]["ps"]
comp_ps  = combined[combined["_group"] == 0]["ps"]

# Common support region
cs_min = max(study_ps.min(), comp_ps.min())
cs_max = min(study_ps.max(), comp_ps.max())

study_in_support = ((study_ps >= cs_min) & (study_ps <= cs_max)).sum()
comp_in_support  = ((comp_ps  >= cs_min) & (comp_ps  <= cs_max)).sum()

overlap_summary = f"""
POSITIVITY CHECK — PS OVERLAP SUMMARY
======================================
Common support region: [{cs_min:.3f}, {cs_max:.3f}]

Study group:
  Total:           {len(study_ps):,}
  In common support: {study_in_support:,} ({study_in_support/len(study_ps)*100:.1f}%)
  PS mean (SD):    {study_ps.mean():.3f} ({study_ps.std():.3f})
  PS range:        [{study_ps.min():.3f}, {study_ps.max():.3f}]
  PS quartiles:    {study_ps.quantile(0.25):.3f} / {study_ps.quantile(0.5):.3f} / {study_ps.quantile(0.75):.3f}

Comparison group:
  Total:           {len(comp_ps):,}
  In common support: {comp_in_support:,} ({comp_in_support/len(comp_ps)*100:.1f}%)
  PS mean (SD):    {comp_ps.mean():.3f} ({comp_ps.std():.3f})
  PS range:        [{comp_ps.min():.3f}, {comp_ps.max():.3f}]
  PS quartiles:    {comp_ps.quantile(0.25):.3f} / {comp_ps.quantile(0.5):.3f} / {comp_ps.quantile(0.75):.3f}

Interpretation:
  {'✔ Good overlap — matching is valid' if study_in_support/len(study_ps) > 0.9 else '⚠ Limited overlap — interpret matched results carefully'}
"""

print(overlap_summary)

with open("ps_overlap_summary.txt", "w") as f:
    f.write(overlap_summary)

# ─────────────────────────────────────────────────────────────
# SMD FUNCTION
# ─────────────────────────────────────────────────────────────

def calculate_smd(df, group_col="_group"):
    """Calculate standardized mean differences for all covariates."""
    results = []
    grp1 = df[df[group_col] == 1]
    grp0 = df[df[group_col] == 0]

    for col in available_covariates:
        if col not in df.columns:
            continue
        m1 = grp1[col].mean()
        m0 = grp0[col].mean()
        s1 = grp1[col].std()
        s0 = grp0[col].std()

        # Use correct SMD formula based on variable type:
        # Binary: Cohen's h denominator sqrt(p_pooled*(1-p_pooled))
        # Continuous: pooled SD denominator
        is_binary = (df[col].nunique(dropna=True) <= 2 and
                     set(np.unique(df[col].dropna())).issubset({0, 1}))
        if is_binary:
            # Standard PS literature SMD for binary variables:
            # pooled SD = sqrt((p1*(1-p1) + p0*(1-p0)) / 2)
            denom = np.sqrt((m1 * (1 - m1) + m0 * (1 - m0)) / 2)
        else:
            denom = np.sqrt((s1**2 + s0**2) / 2)
        # Use 1e-6 threshold to prevent division collapse for rare comorbidities
        smd = abs(m1 - m0) / denom if denom > 1e-6 else 0
        results.append({
            "covariate": col,
            "study_mean":  round(m1, 4),
            "comp_mean":   round(m0, 4),
            "smd":         round(smd, 4),
            "balanced":    smd < 0.1
        })
    return pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────
# MATCHING FUNCTION
# ─────────────────────────────────────────────────────────────

def match_cohorts(combined_df, caliper_sd, label="primary"):
    """
    1:1 nearest-neighbor matching without replacement.
    Caliper applied on logit(PS) scale.
    """
    print(f"\n  Matching ({label}, caliper={caliper_sd} SD)...")

    logit_sd = combined_df["logit_ps"].std()
    caliper  = caliper_sd * logit_sd
    print(f"  Logit PS SD: {logit_sd:.4f} → caliper = {caliper:.4f}")

    study_df = combined_df[combined_df["_group"] == 1].copy().reset_index(drop=True)
    comp_df  = combined_df[combined_df["_group"] == 0].copy().reset_index(drop=True)
    # Explicit index reset ensures no index reuse confusion during greedy matching
    comp_df.index = range(len(comp_df))

    # Shuffle study patients for random matching order
    np.random.seed(RANDOM_SEED)
    study_df = study_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    matched_study_idx = []
    matched_comp_idx  = []
    used_comp         = set()

    for i, study_row in study_df.iterrows():
        s_logit = study_row["logit_ps"]

        # Find eligible comparison patients within caliper
        eligible = comp_df[
            (~comp_df.index.isin(used_comp)) &
            (abs(comp_df["logit_ps"] - s_logit) <= caliper)
        ]

        if len(eligible) == 0:
            continue

        # Select closest match
        closest_idx = (eligible["logit_ps"] - s_logit).abs().idxmin()
        matched_study_idx.append(i)
        matched_comp_idx.append(closest_idx)
        used_comp.add(closest_idx)

    matched_study = study_df.loc[matched_study_idx].copy()
    matched_comp  = comp_df.loc[matched_comp_idx].copy()

    n_matched      = len(matched_study)
    n_unmatched    = len(study_df) - n_matched
    pct_matched    = n_matched / len(study_df) * 100

    print(f"  Matched pairs:    {n_matched:,}")
    print(f"  Unmatched study:  {n_unmatched:,} ({100-pct_matched:.1f}%)")

    # Save matched pairs map for auditability and Step 7 linkage
    matched_pairs = pd.DataFrame({
        "study_patient_id": matched_study["patient_id"].values,
        "comp_patient_id":  matched_comp["patient_id"].values,
        "study_logit_ps":   matched_study["logit_ps"].values,
        "comp_logit_ps":    matched_comp["logit_ps"].values,
        "ps_distance":      abs(matched_study["logit_ps"].values -
                                matched_comp["logit_ps"].values)
    })
    safe_label = "primary" if "primary" in label else "sensitivity"
    matched_pairs.to_csv(f"matched_pairs_{safe_label}.csv", index=False)
    print(f"  Saved: matched_pairs_{safe_label}.csv")

    return matched_study, matched_comp

# ─────────────────────────────────────────────────────────────
# PRIMARY MATCHING (caliper = 0.2 SD)
# ─────────────────────────────────────────────────────────────

matched_study, matched_comp = match_cohorts(
    combined, PRIMARY_CALIPER, label="primary 0.2 SD"
)

# ─────────────────────────────────────────────────────────────
# SENSITIVITY MATCHING (caliper = 0.1 SD)
# ─────────────────────────────────────────────────────────────

matched_study_sens, matched_comp_sens = match_cohorts(
    combined, SENSITIVITY_CALIPER, label="sensitivity 0.1 SD"
)

# ─────────────────────────────────────────────────────────────
# BALANCE TABLE (Sadda eTable 2 equivalent)
# ─────────────────────────────────────────────────────────────

print("\n  Calculating SMDs before and after matching...")

# Before matching
smd_before = calculate_smd(combined)
smd_before = smd_before.rename(columns={
    "study_mean": "study_mean_before",
    "comp_mean":  "comp_mean_before",
    "smd":        "smd_before",
    "balanced":   "balanced_before"
})

# After matching (primary)
matched_combined = pd.concat([matched_study, matched_comp])

# Assert no duplicate comparison patients (no-replacement guarantee)
assert matched_comp["patient_id"].nunique() == len(matched_comp), \
    "Duplicate comparison patients found — check matching logic"

smd_after = calculate_smd(matched_combined)
smd_after = smd_after.rename(columns={
    "study_mean": "study_mean_after",
    "comp_mean":  "comp_mean_after",
    "smd":        "smd_after",
    "balanced":   "balanced_after"
})

balance_table = smd_before.merge(
    smd_after[["covariate", "study_mean_after", "comp_mean_after",
               "smd_after", "balanced_after"]],
    on="covariate"
)

# Summary
n_balanced_before = balance_table["balanced_before"].sum()
n_balanced_after  = balance_table["balanced_after"].sum()
n_total           = len(balance_table)

print(f"\n  Balance summary (primary match):")
print(f"    Before matching: {n_balanced_before}/{n_total} covariates balanced (SMD<0.1)")
print(f"    After matching:  {n_balanced_after}/{n_total} covariates balanced (SMD<0.1)")

# Print imbalanced covariates after matching
imbalanced = balance_table[~balance_table["balanced_after"]]
if len(imbalanced) > 0:
    print(f"\n  ⚠ Still imbalanced after matching (SMD≥0.1):")
    for _, row in imbalanced.iterrows():
        print(f"    {row['covariate']:<30} SMD={row['smd_after']:.3f}")
else:
    print("\n  ✔ All covariates balanced after matching (SMD<0.1)")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 SUMMARY")
print("=" * 60)
print(f"""
  Pre-matching:
    Study group:           {len(study):,}
    Comparison group:      {len(comp):,}

  PRIMARY ANALYSIS (caliper = 0.2 SD):
    Matched pairs:         {len(matched_study):,}
    Study unmatched:       {len(study) - len(matched_study):,}
    Covariates balanced:   {n_balanced_after}/{n_total} (SMD<0.1)

  SENSITIVITY ANALYSIS (caliper = 0.1 SD):
    Matched pairs:         {len(matched_study_sens):,}
    Study unmatched:       {len(study) - len(matched_study_sens):,}
""")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

# Primary matched cohorts — keep patient_id and key clinical vars
keep_cols = ["patient_id", "bariatric_date", "procedure_type",
             "age_at_surgery", "baseline_a1c", "baseline_bmi",
             "dm_duration_years", "ps", "logit_ps", "_group"]

# Assert bariatric_date is present and complete before saving
assert "bariatric_date" in matched_study.columns, "bariatric_date missing from matched_study"
assert matched_study["bariatric_date"].notna().all(), "NaT values in matched_study bariatric_date"
assert "bariatric_date" in matched_comp.columns, "bariatric_date missing from matched_comp"
assert matched_comp["bariatric_date"].notna().all(), "NaT values in matched_comp bariatric_date"

matched_study[keep_cols].to_csv("matched_study.csv", index=False)
matched_comp[keep_cols].to_csv("matched_comparison.csv", index=False)

# Sensitivity matched cohorts
matched_study_sens[keep_cols].to_csv("matched_study_sensitivity.csv", index=False)
matched_comp_sens[keep_cols].to_csv("matched_comparison_sensitivity.csv", index=False)

# Balance table
balance_table.to_csv("balance_table.csv", index=False)

print("  Saved: matched_study.csv")
print("  Saved: matched_comparison.csv")
print("  Saved: matched_study_sensitivity.csv")
print("  Saved: matched_comparison_sensitivity.csv")
print("  Saved: balance_table.csv")
print("  Saved: ps_overlap_summary.txt")
print("\n  Next: run bariatric_step7_a1c_trajectory.py")
