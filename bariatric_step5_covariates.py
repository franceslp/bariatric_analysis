"""
bariatric_step5_covariates.py

Collects all propensity score matching covariates for both cohorts.
Follows Sadda et al. (JAMA Surgery 2026) matching framework adapted
for bariatric surgery glycemic outcomes study.

Covariates collected (year before surgery unless noted):
    Demographics:   age, sex, race, ethnicity
    Glycemic:       baseline A1c, diabetes duration, diabetes type
    Body:           BMI
    Comorbidities:  CKD, CAD, dyslipidemia, hypertension, stroke,
                    heart failure, DM complications (renal, neuro,
                    circulatory, ophthalmic, other)
    Medications:    metformin, any_insulin, rapid_insulin,
                    long_insulin, glp1, sglt2, dpp4,
                    sulfonylurea, tzd
    Procedure:      procedure_type (already in cohort files)

Requires:
    bariatric_study_patients.csv      (from Step 3)
    comparison_group_patients.csv     (from Step 4)

Outputs:
    study_covariates.csv
    comparison_covariates.csv
"""

import pandas as pd
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

BUCKET   = "gs://test-skynet-lh/joseph-sujka/trinetx-gastroparesis-dyspepsia"
CHUNK_SIZE = 100_000

# File paths
DIAG_FILE = f"{BUCKET}/diagnosis.csv"
LAB_FILE  = f"{BUCKET}/lab_result.csv"
VIT_FILE  = f"{BUCKET}/vitals_signs.csv"
MED_FILE  = f"{BUCKET}/medication_ingredient.csv"
PAT_FILE  = f"{BUCKET}/patient.csv"

# A1c LOINC codes
A1C_LOINC = {"4548-4", "17856-6", "4549-2"}

# BMI LOINC
BMI_LOINC = {"39156-5"}

# Diagnosis codes for comorbidities (Sadda eTable 1)
# Comorbidity definitions exactly per Sadda et al. (JAMA Surgery 2026, eTable 1)
# No expansion of ICD granularity beyond published definitions.
# If reviewers ask why definitions are narrow, cite Sadda replication fidelity.
# Window: 365 days pre-op, consistent with Sadda's "year before index date."
DIAG_CODES = {
    "dm_renal":      lambda c: c.startswith("E10.2") or c.startswith("E11.2"),
    "dm_neuro":      lambda c: c.startswith("E10.4") or c.startswith("E11.4"),
    "dm_circ":       lambda c: c.startswith("E10.5") or c.startswith("E11.5"),
    "dm_opthal":     lambda c: c.startswith("E10.3") or c.startswith("E11.3"),
    "dm_other":      lambda c: c.startswith("E10.6") or c.startswith("E11.6"),
    "dyslipidemia":  lambda c: c.startswith("E78"),
    "ckd":           lambda c: c.startswith("N18"),
    "stroke":        lambda c: c.startswith("I63"),
    "cad":           lambda c: c.startswith("I25.1"),
    "heart_failure": lambda c: c.startswith("I50"),
    "hypertension":  lambda c: c == "I10",
}

# FIX Major 1: diabetes type uses lifetime history before surgery (not window)
DM_TYPE_CODES = {
    "t1dm": lambda c: c.startswith("E10"),
    "t2dm": lambda c: c.startswith("E11"),
}

# NOTE: Continuous covariates (age, A1c, BMI, dm_duration_years) are output
# in raw form. Standardization (mean=0, SD=1) will be applied in Step 6
# propensity model estimation to ensure PS logistic regression stability.

# Medication RxNorm ingredient codes
MED_CODES = {
    "metformin":     {"6809"},
    "rapid_insulin": {"51428", "86009", "311036", "1156706"},
    "long_insulin":  {"253182", "274783", "1151131", "2200801"},
    "glp1":          {"60548", "475968", "2200644", "1991302", "1440051"},
    "sglt2":         {"1488574", "1545653", "1602111", "1932591"},
    "dpp4":          {"593411", "593533", "1100699", "884220"},
    "sulfonylurea":  {"4815", "4821", "25789"},
    "tzd":           {"33738", "84108"},
}

WINDOW_DAYS = 365   # year before surgery

def parse_dates(series):
    series = series.fillna("").astype(str)
    result = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    mask = result.isna() & (series.str.strip() != "")
    if mask.any():
        result.loc[mask] = pd.to_datetime(
            series.loc[mask], format="mixed", dayfirst=False, errors="coerce"
        )
    return result

def in_window(event_date, surgery_date, days=WINDOW_DAYS):
    """Check if event_date is within WINDOW_DAYS before surgery_date."""
    diff = (surgery_date - event_date).days
    return 0 <= diff <= days

# ─────────────────────────────────────────────────────────────
# LOAD COHORTS
# ─────────────────────────────────────────────────────────────

import datetime
print("=" * 60)
print("STEP 5: COLLECTING MATCHING COVARIATES")
print(f"  Run date:   {datetime.date.today()}")
print(f"  Dataset:    {BUCKET}")
print(f"  Sadda ref:  JAMA Surgery 2026, doi:10.1001/jamasurg.2026.1593")
print("=" * 60)

study = pd.read_csv("bariatric_study_patients.csv", dtype=str)
comp  = pd.read_csv("comparison_group_patients.csv", dtype=str)

for df in [study, comp]:
    df["bariatric_date"] = parse_dates(df["bariatric_date"])
    df["first_dm_date"]  = parse_dates(df["first_dm_date"])

# Combine both cohorts for single-pass streaming
study["cohort"] = "study"
comp["cohort"]  = "comparison"

# Align columns — study has first_gp_date, comp does not
if "first_gp_date" not in comp.columns:
    comp["first_gp_date"] = pd.NaT
if "first_gp_date" not in study.columns:
    study["first_gp_date"] = pd.NaT

study["first_gp_date"] = parse_dates(study["first_gp_date"].astype(str))

# Ensure post_op_gp_flag exists in both — study group won't have it
if "post_op_gp_flag" not in study.columns:
    study["post_op_gp_flag"] = False
if "post_op_gp_flag" not in comp.columns:
    comp["post_op_gp_flag"] = False

all_patients = pd.concat([
    study[["patient_id", "bariatric_date", "first_dm_date",
           "first_gp_date", "age_at_surgery", "procedure_type",
           "post_op_gp_flag", "cohort"]],
    comp[["patient_id", "bariatric_date", "first_dm_date",
          "first_gp_date", "age_at_surgery", "procedure_type",
          "post_op_gp_flag", "cohort"]]
], ignore_index=True)

# PRIMARY ANALYSIS: first qualifying surgery only (Option 2)
# Patients appearing in both cohorts had two bariatric surgeries
# (typically sleeve first, then bypass after developing gastroparesis).
# For the primary analysis these patients are excluded from BOTH cohorts
# to prevent overlap and avoid immortal time / selection bias.
# These patients are saved separately as a clinically meaningful
# sensitivity analysis population (sleeve → GP → revision bypass pathway).
overlap = set(study["patient_id"]) & set(comp["patient_id"])
if len(overlap) > 0:
    print(f"  NOTE: {len(overlap):,} patients appear in both cohorts")
    print(f"  (sleeve→gastroparesis→revision bypass pathway)")
    print(f"  Excluding from both cohorts for primary analysis.")
    print(f"  Saved separately as revision_pathway_patients.csv")
    revision = pd.concat([
        study[study["patient_id"].isin(overlap)],
        comp[comp["patient_id"].isin(overlap)]
    ]).sort_values(["patient_id", "bariatric_date"])
    revision.to_csv("revision_pathway_patients.csv", index=False)
    study = study[~study["patient_id"].isin(overlap)].copy()
    comp  = comp[~comp["patient_id"].isin(overlap)].copy()

print(f"  Study group (primary analysis):      {len(study):,}")
print(f"  Comparison group (primary analysis): {len(comp):,}")

# Rebuild all_patients after overlap removal
all_patients = pd.concat([
    study[["patient_id", "bariatric_date", "first_dm_date",
           "first_gp_date", "age_at_surgery", "procedure_type",
           "post_op_gp_flag", "cohort"]],
    comp[["patient_id", "bariatric_date", "first_dm_date",
          "first_gp_date", "age_at_surgery", "procedure_type",
          "post_op_gp_flag", "cohort"]]
], ignore_index=True)

# Assert one surgery date per patient — now safe after overlap removal
max_dates = all_patients.groupby("patient_id")["bariatric_date"].nunique().max()
assert max_dates == 1, f"Duplicate surgery dates remain after overlap removal (max={max_dates})"

# Index date validation
n_missing_surg = all_patients["bariatric_date"].isna().sum()
assert n_missing_surg == 0, f"{n_missing_surg} patients have missing surgery date"

n_missing_dm = all_patients["first_dm_date"].isna().sum()
if n_missing_dm > 0:
    print(f"  WARNING: {n_missing_dm:,} patients have missing DM date — duration will be NaN")

bad_timing = (all_patients["bariatric_date"] < all_patients["first_dm_date"]).sum()
if bad_timing > 0:
    print(f"  WARNING: {bad_timing:,} patients have surgery before DM date — check Step 3/4")

# Fast lookups
surgery_dict = dict(zip(all_patients["patient_id"], all_patients["bariatric_date"]))
dm_dict      = dict(zip(all_patients["patient_id"], all_patients["first_dm_date"]))
all_ids      = set(all_patients["patient_id"].unique())
print(f"  Total patients for covariate collection: {len(all_ids):,}")

# ─────────────────────────────────────────────────────────────
# CALCULATE DIABETES DURATION
# ─────────────────────────────────────────────────────────────

all_patients["dm_duration_years"] = (
    (all_patients["bariatric_date"] - all_patients["first_dm_date"]).dt.days / 365.25
).round(2)

# Winsorize: cap at 0-50 years; negative or implausible values set to NaN
all_patients["dm_duration_years"] = all_patients["dm_duration_years"].where(
    (all_patients["dm_duration_years"] >= 0) &
    (all_patients["dm_duration_years"] <= 50),
    other=float("nan")
)

print(f"\n  Diabetes duration calculated for all patients")
n_implausible = all_patients["dm_duration_years"].isna().sum()
if n_implausible > 0:
    print(f"  WARNING: {n_implausible:,} patients with implausible DM duration set to NaN")

# ─────────────────────────────────────────────────────────────
# LOAD PATIENT FILE — sex, race, ethnicity
# ─────────────────────────────────────────────────────────────

print("\n  Loading patient.csv...")

proc_pat = subprocess.Popen(["gsutil", "cat", PAT_FILE], stdout=subprocess.PIPE)
if proc_pat.stdout is None:
    raise RuntimeError("Failed to open patient.csv")

pat_df = pd.read_csv(proc_pat.stdout, dtype=str, low_memory=False)
proc_pat.stdout.close()
proc_pat.wait()

pat_df.columns = pat_df.columns.str.strip().str.lower()
pat_df["patient_id"] = pat_df["patient_id"].fillna("").astype(str)
pat_df = pat_df[pat_df["patient_id"].isin(all_ids)][
    ["patient_id", "sex", "race", "ethnicity"]
]
print(f"  Demographics loaded for {len(pat_df):,} patients")

# ─────────────────────────────────────────────────────────────
# PASS 1: STREAM DIAGNOSIS FILE — comorbidities in window
# ─────────────────────────────────────────────────────────────

print("\n  Pass 1: Streaming diagnosis.csv — comorbidities...")

proc_diag = subprocess.Popen(["gsutil", "cat", DIAG_FILE], stdout=subprocess.PIPE)
if proc_diag.stdout is None:
    raise RuntimeError("Failed to open diagnosis.csv")

# patient_id -> set of comorbidity flags (window-based)
comorbidity_flags = defaultdict(set)
# patient_id -> diabetes type (lifetime history before surgery)
dm_type_flags = defaultdict(set)
total_rows = 0
chunk_num  = 0

for chunk in pd.read_csv(proc_diag.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows += len(chunk)
    chunk_num  += 1

    chunk = chunk[chunk["patient_id"].isin(all_ids)].copy()
    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    for row in chunk.itertuples(index=False):
        pid     = row.patient_id
        surg_dt = surgery_dict.get(pid)
        if surg_dt is None:
            continue

        # Diabetes type: lifetime history before surgery (not window-restricted)
        # Must come BEFORE the window filter so pre-2016 diagnoses are captured
        if row.date < surg_dt:
            for flag, condition in DM_TYPE_CODES.items():
                if condition(row.code):
                    dm_type_flags[pid].add(flag)

        # Comorbidities: window-restricted (365 days pre-op) per Sadda et al.
        if not in_window(row.date, surg_dt):
            continue
        for flag, condition in DIAG_CODES.items():
            if condition(row.code):
                comorbidity_flags[pid].add(flag)

    if chunk_num % 50 == 0:
        print(f"  Processed {total_rows:,} diagnosis rows...")

proc_diag.stdout.close()
proc_diag.wait()
print(f"  Total diagnosis rows: {total_rows:,}")

# ─────────────────────────────────────────────────────────────
# PASS 2: STREAM LAB FILE — baseline A1c
# ─────────────────────────────────────────────────────────────

print("\n  Pass 2: Streaming lab_result.csv — baseline A1c...")

proc_lab = subprocess.Popen(["gsutil", "cat", LAB_FILE], stdout=subprocess.PIPE)
if proc_lab.stdout is None:
    raise RuntimeError("Failed to open lab_result.csv")

# patient_id -> closest A1c before surgery
baseline_a1c = {}
total_rows2  = 0
chunk_num2   = 0

for chunk in pd.read_csv(proc_lab.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows2 += len(chunk)
    chunk_num2  += 1

    chunk = chunk[
        chunk["patient_id"].isin(all_ids) &
        chunk["code"].isin(A1C_LOINC)
    ].copy()

    if chunk.empty:
        continue

    chunk["date"] = parse_dates(chunk["date"])
    chunk = chunk.dropna(subset=["date"])

    # Keep numeric values only
    chunk["lab_result_num_val"] = pd.to_numeric(
        chunk["lab_result_num_val"], errors="coerce"
    )
    chunk = chunk.dropna(subset=["lab_result_num_val"])

    # Physiologic filter
    chunk = chunk[
        (chunk["lab_result_num_val"] >= 3) &
        (chunk["lab_result_num_val"] <= 20)
    ]

    for row in chunk.itertuples(index=False):
        pid     = row.patient_id
        surg_dt = surgery_dict.get(pid)
        if surg_dt is None:
            continue
        if not in_window(row.date, surg_dt):
            continue
        # Keep closest A1c to surgery (smallest absolute time difference)
        diff = abs((surg_dt - row.date).days)
        if pid not in baseline_a1c or diff < baseline_a1c[pid][2]:
            baseline_a1c[pid] = (row.date, row.lab_result_num_val, diff)

    if chunk_num2 % 50 == 0:
        print(f"  Processed {total_rows2:,} lab rows...")

proc_lab.stdout.close()
proc_lab.wait()
print(f"  Total lab rows: {total_rows2:,}")
print(f"  Patients with baseline A1c: {len(baseline_a1c):,}")

# ─────────────────────────────────────────────────────────────
# PASS 3: STREAM VITALS FILE — baseline BMI
# ─────────────────────────────────────────────────────────────

print("\n  Pass 3: Streaming vitals_signs.csv — baseline BMI...")

proc_vit = subprocess.Popen(["gsutil", "cat", VIT_FILE], stdout=subprocess.PIPE)
if proc_vit.stdout is None:
    raise RuntimeError("Failed to open vitals_signs.csv")

baseline_bmi = {}
total_rows3  = 0
chunk_num3   = 0

for chunk in pd.read_csv(proc_vit.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows3 += len(chunk)
    chunk_num3  += 1

    chunk = chunk[
        chunk["patient_id"].isin(all_ids) &
        chunk["code"].isin(BMI_LOINC)
    ].copy()

    if chunk.empty:
        continue

    chunk["date"]  = parse_dates(chunk["date"])
    chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
    chunk = chunk.dropna(subset=["date", "value"])

    # Physiologic BMI filter
    chunk = chunk[
        (chunk["value"] >= 10) &
        (chunk["value"] <= 100)
    ]

    for row in chunk.itertuples(index=False):
        pid     = row.patient_id
        surg_dt = surgery_dict.get(pid)
        if surg_dt is None:
            continue
        if not in_window(row.date, surg_dt):
            continue
        # Keep closest BMI to surgery (smallest absolute time difference)
        diff = abs((surg_dt - row.date).days)
        if pid not in baseline_bmi or diff < baseline_bmi[pid][2]:
            baseline_bmi[pid] = (row.date, row.value, diff)

    if chunk_num3 % 50 == 0:
        print(f"  Processed {total_rows3:,} vitals rows...")

proc_vit.stdout.close()
proc_vit.wait()
print(f"  Total vitals rows: {total_rows3:,}")
print(f"  Patients with baseline BMI: {len(baseline_bmi):,}")

# ─────────────────────────────────────────────────────────────
# PASS 4: STREAM MEDICATION FILE — drug flags in window
# ─────────────────────────────────────────────────────────────

print("\n  Pass 4: Streaming medication_ingredient.csv — medications...")

proc_med = subprocess.Popen(["gsutil", "cat", MED_FILE], stdout=subprocess.PIPE)
if proc_med.stdout is None:
    raise RuntimeError("Failed to open medication_ingredient.csv")

# patient_id -> set of medication flags
med_flags  = defaultdict(set)
total_rows4 = 0
chunk_num4  = 0

# Build reverse lookup: code -> drug class
code_to_class = {}
for drug_class, codes in MED_CODES.items():
    for code in codes:
        code_to_class[code] = drug_class

for chunk in pd.read_csv(proc_med.stdout, dtype=str, chunksize=CHUNK_SIZE, low_memory=False):
    if chunk.empty:
        continue

    chunk.columns = chunk.columns.str.strip().str.lower()
    chunk["patient_id"] = chunk["patient_id"].fillna("").astype(str)
    chunk["code"]       = chunk["code"].fillna("").astype(str).str.strip()

    total_rows4 += len(chunk)
    chunk_num4  += 1

    chunk = chunk[
        chunk["patient_id"].isin(all_ids) &
        chunk["code"].isin(code_to_class)
    ].copy()

    if chunk.empty:
        continue

    chunk["start_date"] = parse_dates(chunk["start_date"])
    chunk = chunk.dropna(subset=["start_date"])

    for row in chunk.itertuples(index=False):
        pid     = row.patient_id
        surg_dt = surgery_dict.get(pid)
        if surg_dt is None:
            continue
        # Medication exposure: lifetime pre-op use (start_date < surgery_date)
        # Rationale: chronic diabetes medications (metformin, insulin, GLP-1)
        # are often initiated years before surgery and reflect disease severity
        # at time of bariatric intervention, not just recent prescribing patterns.
        # This is consistent with Sadda-style binary PS covariates where
        # medication history acts as a disease-severity marker.
        # Methods note: justify as "ever-use prior to index date" in manuscript.
        if row.start_date >= surg_dt:
            continue
        drug_class = code_to_class.get(row.code)
        if drug_class:
            med_flags[pid].add(drug_class)

    if chunk_num4 % 50 == 0:
        print(f"  Processed {total_rows4:,} medication rows...")

proc_med.stdout.close()
proc_med.wait()
print(f"  Total medication rows: {total_rows4:,}")

# ─────────────────────────────────────────────────────────────
# ASSEMBLE COVARIATES
# ─────────────────────────────────────────────────────────────

print("\n  Assembling covariates...")

rows = []
for _, pt in all_patients.iterrows():
    pid     = pt["patient_id"]
    surg_dt = pt["bariatric_date"]
    flags   = comorbidity_flags.get(pid, set())
    meds    = med_flags.get(pid, set())
    a1c     = baseline_a1c.get(pid)
    bmi     = baseline_bmi.get(pid)

    # Any insulin flag
    any_insulin = (
        "rapid_insulin" in meds or "long_insulin" in meds
    )

    row = {
        "patient_id":        pid,
        "cohort":            pt["cohort"],
        # post_op_gp_flag excluded from covariates — outcome-adjacent variable
        # kept in comparison_group_patients.csv for descriptive use only
        "age_at_surgery":    pt["age_at_surgery"],
        "procedure_type":    pt["procedure_type"],
        "dm_duration_years":   pt["dm_duration_years"],
        "dm_duration_missing": 1 if pd.isna(pt["dm_duration_years"]) else 0,
        "baseline_a1c":         a1c[1] if a1c else None,
        "baseline_a1c_missing": 0 if a1c else 1,
        "baseline_bmi":         bmi[1] if bmi else None,
        "baseline_bmi_missing": 0 if bmi else 1,
        # Demographics (filled from patient.csv below)
        "sex":               None,
        "race":              None,
        "ethnicity":         None,
        # FIX Major 1: diabetes type from lifetime history (not window)
        "t1dm":              "t1dm" in dm_type_flags.get(pid, set()),
        "t2dm":              "t2dm" in dm_type_flags.get(pid, set()),
        # DM complications
        "dm_renal":          "dm_renal" in flags,
        "dm_neuro":          "dm_neuro" in flags,
        "dm_circ":           "dm_circ" in flags,
        "dm_opthal":         "dm_opthal" in flags,
        "dm_other":          "dm_other" in flags,
        # Comorbidities
        "dyslipidemia":      "dyslipidemia" in flags,
        "ckd":               "ckd" in flags,
        "stroke":            "stroke" in flags,
        "cad":               "cad" in flags,
        "heart_failure":     "heart_failure" in flags,
        "hypertension":      "hypertension" in flags,
        # Medications
        "metformin":         "metformin" in meds,
        "any_insulin":       any_insulin,
        "rapid_insulin":     "rapid_insulin" in meds,
        "long_insulin":      "long_insulin" in meds,
        "glp1":              "glp1" in meds,
        "sglt2":             "sglt2" in meds,
        "dpp4":              "dpp4" in meds,
        "sulfonylurea":      "sulfonylurea" in meds,
        "tzd":               "tzd" in meds,
    }
    rows.append(row)

cov_df = pd.DataFrame(rows)

# Merge demographics
cov_df = cov_df.merge(pat_df, on="patient_id", how="left", suffixes=("", "_pat"))
for col in ["sex", "race", "ethnicity"]:
    if f"{col}_pat" in cov_df.columns:
        cov_df[col] = cov_df[f"{col}_pat"]
        cov_df.drop(columns=[f"{col}_pat"], inplace=True)

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 SUMMARY")
print("=" * 60)

for cohort_name in ["study", "comparison"]:
    sub = cov_df[cov_df["cohort"] == cohort_name]
    print(f"\n  {cohort_name.upper()} GROUP (n={len(sub):,})")
    print(f"    Baseline A1c available:    {sub['baseline_a1c'].notna().sum():,} ({sub['baseline_a1c'].notna().mean()*100:.1f}%)")
    print(f"    Baseline BMI available:    {sub['baseline_bmi'].notna().sum():,} ({sub['baseline_bmi'].notna().mean()*100:.1f}%)")
    print(f"    Mean baseline A1c:         {sub['baseline_a1c'].mean():.2f}%")
    print(f"    Mean baseline BMI:         {sub['baseline_bmi'].mean():.1f} kg/m2")
    print(f"    Mean DM duration:          {sub['dm_duration_years'].mean():.1f} years")
    print(f"    Metformin use:             {sub['metformin'].sum():,} ({sub['metformin'].mean()*100:.1f}%)")
    print(f"    Any insulin use:           {sub['any_insulin'].sum():,} ({sub['any_insulin'].mean()*100:.1f}%)")
    print(f"    GLP-1 use:                 {sub['glp1'].sum():,} ({sub['glp1'].mean()*100:.1f}%)")
    print(f"    Hypertension:              {sub['hypertension'].sum():,} ({sub['hypertension'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────

# Missingness summary table — useful for supplement
print("\n  Missingness summary (% missing per covariate):")
miss_cols = ["baseline_a1c", "baseline_bmi", "dm_duration_years", "sex", "race"]
for cohort_name in ["study", "comparison"]:
    sub = cov_df[cov_df["cohort"] == cohort_name]
    print(f"\n    {cohort_name.upper()} GROUP:")
    for col in miss_cols:
        if col in sub.columns:
            pct = sub[col].isna().mean() * 100
            print(f"      {col:<25} {pct:.1f}% missing")

# Save missingness summary to CSV for supplement
miss_rows = []
for cohort_name in ["study", "comparison"]:
    sub = cov_df[cov_df["cohort"] == cohort_name]
    for col in cov_df.columns:
        if col in ["patient_id", "cohort"]:
            continue
        pct = sub[col].isna().mean() * 100
        miss_rows.append({
            "cohort": cohort_name,
            "covariate": col,
            "pct_missing": round(pct, 1)
        })
pd.DataFrame(miss_rows).to_csv("covariate_missingness.csv", index=False)
print("\n  Saved: covariate_missingness.csv")

study_cov = cov_df[cov_df["cohort"] == "study"].drop(columns=["cohort"])
comp_cov  = cov_df[cov_df["cohort"] == "comparison"].drop(columns=["cohort"])

study_cov.to_csv("study_covariates.csv", index=False)
comp_cov.to_csv("comparison_covariates.csv", index=False)

print(f"\n  Saved: study_covariates.csv ({len(study_cov):,} patients)")
print(f"  Saved: comparison_covariates.csv ({len(comp_cov):,} patients)")
print("  Next: run bariatric_step6_propensity_matching.py")
