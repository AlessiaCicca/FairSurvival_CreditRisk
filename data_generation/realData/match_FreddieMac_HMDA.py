"""
data_generation/fnma/match_hmda.py

Match Freddie Mac origination data with HMDA demographic data.
Produces matched_{YEAR}.csv with Freddie variables + HMDA demographics.

Usage:
    python match_hmda.py --drive_root /path/to/thesis_data --year 2024

    # Combine all matched years into one file:
    python match_hmda.py --drive_root /path/to/thesis_data --year 2024 --combine

Expected directory structure:
    drive_root/
        hmda/
            hmda_2024.csv
        freddie/
            historical_data_2024/
                historical_data_2024Q1.zip
                historical_data_2024Q2.zip
                ...
        output/                        <- created automatically

Output:
    drive_root/output/matched_{YEAR}.csv
"""

import os
import gc
import glob
import shutil
import zipfile
import argparse

import numpy as np
import pandas as pd


# ── Column layout ─────────────────────────────────────────────────────────────

FREDDIE_ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer",
    "maturity_date", "msa", "mi_pct", "num_units", "occupancy_status",
    "original_cltv", "original_dti", "original_upb", "original_ltv",
    "original_interest_rate", "channel", "ppm_flag", "amortization_type",
    "property_state", "property_type", "postal_code", "loan_sequence_number",
    "loan_purpose", "original_loan_term", "num_borrowers", "seller_name",
    "servicer_name", "super_conforming_flag", "pre_relief_refi_seq",
    "special_eligibility", "relief_refi_indicator", "property_valuation",
    "io_indicator", "mi_cancellation",
]

FREDDIE_OCC_MAP     = {"P": 1, "S": 2, "I": 3}
FREDDIE_PURPOSE_MAP = {"P": 1, "C": 32, "N": 31, "R": 31}

MATCH_KEYS = [
    "state_code", "msa", "loan_amount_r", "interest_rate",
    "loan_term", "num_units", "occupancy_type", "loan_purpose",
]

HMDA_DEMO_COLS = [
    "derived_race",
    "applicant_race_1", "applicant_race_2", "applicant_race_3",
    "applicant_race_4", "applicant_race_5",
    "co_applicant_race_1", "co_applicant_race_2", "co_applicant_race_3",
    "co_applicant_race_4", "co_applicant_race_5",
    "derived_sex", "applicant_sex", "co_applicant_sex",
    "applicant_age", "co_applicant_age",
    "applicant_age_above_62", "co_applicant_age_above_62",
]

HMDA_EXTRA_COLS = [
    "activity_year", "lei", "action_taken", "purchaser_type", "loan_type",
    "property_type", "lien_status", "reverse_mortgage",
    "open_end_line_of_credit", "business_or_commercial",
    "conforming_loan_limit", "derived_loan_product_type",
    "derived_dwelling_category", "county_code", "census_tract",
    "applicant_ethnicity_1", "co_applicant_ethnicity_1", "derived_ethnicity",
    "income", "rate_spread", "hoepa_status", "total_loan_costs",
    "origination_charges", "discount_points", "lender_credits",
    "loan_to_value_ratio", "intro_rate_period", "negative_amortization",
    "interest_only_payment", "balloon_payment",
    "other_nonamortizing_features", "property_value",
    "manufactured_home_secured_property_type",
    "manufactured_home_land_property_interest",
    "submission_of_application", "initially_payable_to_institution",
    "aus_1", "denial_reason_1", "tract_population",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income", "tract_to_msa_income_percentage",
    "tract_owner_occupied_units", "tract_one_to_four_family_homes",
    "tract_median_age_of_housing_units",
]

CHUNK_SIZE    = 200_000
KEEP_ALL_HMDA = False


# ── Freddie utilities ─────────────────────────────────────────────────────────

def unzip_freddie_year(year: int, freddie_dir: str,
                       freddie_local: str) -> list:
    """
    Find Freddie Mac quarterly zip files and extract origination .txt files
    to local disk. Returns sorted list of extracted .txt paths.
    """
    zip_pattern_sub  = os.path.join(
        freddie_dir, f"historical_data_{year}", f"historical_data_{year}Q*.zip"
    )
    zip_pattern_flat = os.path.join(
        freddie_dir, f"historical_data_{year}Q*.zip"
    )
    zip_files = glob.glob(zip_pattern_sub) or glob.glob(zip_pattern_flat)

    if not zip_files:
        raise FileNotFoundError(
            f"\nNo zip files found for {year}.\n"
            f"Patterns checked:\n  {zip_pattern_sub}\n  {zip_pattern_flat}"
        )

    print(f"Found {len(zip_files)} zip files for {year}:")
    extracted = []

    for zpath in sorted(zip_files):
        zname    = os.path.basename(zpath)
        txt_dest = os.path.join(freddie_local,
                                zname.replace(".zip", ".txt"))

        if os.path.exists(txt_dest):
            size_mb = os.path.getsize(txt_dest) / 1e6
            print(f"  {zname} -> already extracted ({size_mb:.0f} MB), skip")
            extracted.append(txt_dest)
            continue

        print(f"  Extracting {zname} ...", end=" ", flush=True)
        with zipfile.ZipFile(zpath, "r") as z:
            orig_files = [f for f in z.namelist()
                          if "time" not in f.lower() and f.endswith(".txt")]
            if not orig_files:
                print(f"\n  WARNING: no origination file in {zname}")
                continue
            with z.open(orig_files[0]) as src, open(txt_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

        size_mb = os.path.getsize(txt_dest) / 1e6
        print(f"OK ({size_mb:.0f} MB)")
        extracted.append(txt_dest)

    return sorted(extracted)


def load_and_prepare_freddie(txt_files: list, year: int) -> pd.DataFrame:
    """Load all Freddie .txt files, apply mappings, rename columns for merge."""
    frames = []
    for f in txt_files:
        df = pd.read_csv(f, sep="|", header=None, names=FREDDIE_ORIG_COLS,
                         usecols=None, dtype=str, low_memory=False)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")
        frames.append(df)

    freddie = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"Total loaded: {len(freddie):,} rows")

    # Filter by year
    freddie["fp_year"] = freddie["first_payment_date"].str[:4]
    freddie = freddie[freddie["fp_year"].isin([str(year), str(year - 1)])]
    print(f"After year filter: {len(freddie):,} rows")

    # Save original codes before mapping
    freddie["occupancy_status_orig"] = freddie["occupancy_status"].copy()
    freddie["loan_purpose_orig"]     = freddie["loan_purpose"].copy()

    # zip3
    freddie["zip3"] = freddie["postal_code"].str[:3]

    # Apply mappings
    freddie["occupancy_type"] = freddie["occupancy_status"].map(FREDDIE_OCC_MAP)
    freddie["loan_purpose"]   = freddie["loan_purpose"].map(FREDDIE_PURPOSE_MAP)

    # Rename to HMDA key names
    freddie.rename(columns={
        "property_state":         "state_code",
        "original_upb":           "loan_amount_r",
        "original_interest_rate": "interest_rate",
        "original_loan_term":     "loan_term",
    }, inplace=True)

    # Numeric conversions
    for col in ["loan_amount_r", "interest_rate", "loan_term", "num_units",
                "original_ltv", "original_cltv", "original_dti",
                "credit_score", "mi_pct"]:
        if col in freddie.columns:
            freddie[col] = pd.to_numeric(freddie[col], errors="coerce")

    freddie["msa"]        = freddie["msa"].str.strip().fillna("")
    freddie["state_code"] = freddie["state_code"].str.strip()

    before = len(freddie)
    freddie.dropna(subset=MATCH_KEYS, inplace=True)
    print(f"After dropna on keys: {len(freddie):,} rows "
          f"(removed {before - len(freddie):,})")

    return freddie


# ── HMDA utilities ────────────────────────────────────────────────────────────

def find_hmda_file(year: int, hmda_dir: str) -> str:
    candidates = [
        os.path.join(hmda_dir, f"hmda_{year}.csv"),
        os.path.join(hmda_dir, f"{year}_public_lar.csv"),
        os.path.join(hmda_dir, f"hmda_{year}_nationwide_all-records_labels.csv"),
        os.path.join(hmda_dir, f"hmda_{year}_nationwide_all-records_codes.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    found = glob.glob(os.path.join(hmda_dir, f"*{year}*.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(
        f"\nNo HMDA file found for {year} in '{hmda_dir}'.\n"
        f"Expected: hmda_{year}.csv"
    )


def prepare_hmda_chunk(chunk: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter and prepare one HMDA chunk for merging."""
    chunk.rename(columns=lambda c: c.replace("-", "_"), inplace=True)

    if "activity_year" in chunk.columns:
        chunk = chunk[chunk["activity_year"].astype(str) == str(year)]
    if chunk.empty:
        return chunk

    chunk = chunk[chunk["purchaser_type"].astype(str) == "3"]
    if chunk.empty:
        return chunk

    chunk = chunk[chunk["action_taken"].astype(str) == "1"]
    if chunk.empty:
        return chunk

    chunk = chunk[chunk["total_units"].isin(["1", "2", "3", "4"])]
    if chunk.empty:
        return chunk

    for col in ["loan_amount", "interest_rate", "loan_term",
                "total_units", "occupancy_type", "loan_purpose"]:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk = chunk[chunk["loan_purpose"].isin([1, 31, 32])]
    if chunk.empty:
        return chunk

    chunk["loan_amount_r"] = (chunk["loan_amount"] / 1000).round() * 1000

    rename_map = {}
    if "total_units"    in chunk.columns: rename_map["total_units"]    = "num_units"
    if "derived_msa_md" in chunk.columns: rename_map["derived_msa_md"] = "msa"
    chunk.rename(columns=rename_map, inplace=True)

    if "state_code" in chunk.columns:
        chunk["state_code"] = chunk["state_code"].str.strip()
    if "msa" in chunk.columns:
        chunk["msa"] = chunk["msa"].str.strip().fillna("")

    chunk.dropna(subset=MATCH_KEYS, inplace=True)
    if chunk.empty:
        return chunk

    if KEEP_ALL_HMDA:
        return chunk

    cols_to_keep = (
        MATCH_KEYS +
        ["loan_amount"] +
        [c for c in HMDA_DEMO_COLS  if c in chunk.columns] +
        [c for c in HMDA_EXTRA_COLS if c in chunk.columns]
    )
    seen = set()
    cols_to_keep = [c for c in cols_to_keep
                    if c in chunk.columns and not (c in seen or seen.add(c))]
    return chunk[cols_to_keep]


# ── Main match logic ──────────────────────────────────────────────────────────

def run_match(year: int, drive_root: str) -> None:
    hmda_dir      = os.path.join(drive_root, "hmda")
    freddie_dir   = os.path.join(drive_root, "freddie")
    freddie_local = os.path.join(drive_root, "freddie_local_tmp")
    output_dir    = os.path.join(drive_root, "output")

    os.makedirs(output_dir,    exist_ok=True)
    os.makedirs(freddie_local, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Year       : {year}")
    print(f"  HMDA dir   : {hmda_dir}")
    print(f"  Freddie dir: {freddie_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: extract and load Freddie ─────────────────────────────────────
    print(f"Extracting Freddie Mac {year}...")
    txt_files     = unzip_freddie_year(year, freddie_dir, freddie_local)
    freddie_ready = load_and_prepare_freddie(txt_files, year)
    print(f"\nFreddie ready: {len(freddie_ready):,} rows, "
          f"{len(freddie_ready.columns)} columns\n")

    # ── Step 2: chunk match with HMDA ────────────────────────────────────────
    hmda_path = find_hmda_file(year, hmda_dir)
    print(f"HMDA file  : {hmda_path}")
    print(f"Chunk size : {CHUNK_SIZE:,}\n")

    chunk_results  = []
    chunk_num      = 0
    total_rows     = 0
    total_filtered = 0
    total_matched  = 0

    for chunk in pd.read_csv(hmda_path, dtype=str,
                             chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num  += 1
        total_rows += len(chunk)

        chunk_clean = prepare_hmda_chunk(chunk, year)
        del chunk
        gc.collect()

        if chunk_clean.empty:
            if chunk_num % 10 == 0:
                print(f"  Chunk {chunk_num:3d} | read {total_rows:>10,} | "
                      f"filtered {total_filtered:>8,} | matched {total_matched:>7,}")
            continue

        total_filtered += len(chunk_clean)

        matched = pd.merge(chunk_clean, freddie_ready, on=MATCH_KEYS,
                           how="inner", suffixes=("_hmda", "_freddie"))
        del chunk_clean
        gc.collect()

        if not matched.empty:
            total_matched += len(matched)
            chunk_results.append(matched)

        if chunk_num % 10 == 0:
            print(f"  Chunk {chunk_num:3d} | read {total_rows:>10,} | "
                  f"filtered {total_filtered:>8,} | matched {total_matched:>7,}")

    print(f"\nDone. Chunks: {chunk_num:,} | HMDA rows: {total_rows:,} | "
          f"Filtered: {total_filtered:,} | Raw matches: {total_matched:,}")

    if not chunk_results:
        print("No matches found. Check paths and year.")
        return

    # ── Step 3: deduplicate and save ─────────────────────────────────────────
    print("\nCombining chunks...")
    all_matches = pd.concat(chunk_results, ignore_index=True)
    del chunk_results
    gc.collect()
    print(f"Total matches before dedup: {len(all_matches):,}")

    dup_freddie  = all_matches.duplicated(subset=["loan_sequence_number"], keep=False)
    dup_hmda     = all_matches.duplicated(subset=MATCH_KEYS, keep=False)
    is_ambiguous = dup_freddie | dup_hmda
    n_amb        = is_ambiguous.sum()

    print(f"Collisions removed : {n_amb:,}  ({100*n_amb/len(all_matches):.1f}%)")
    print(f"  Freddie dupes    : {dup_freddie.sum():,}")
    print(f"  HMDA dupes       : {dup_hmda.sum():,}")

    final = all_matches[~is_ambiguous].copy()
    del all_matches
    gc.collect()

    n_clean = len(final)
    print(f"Clean 1-to-1 matches: {n_clean:,}")
    print(f"Match rate: {n_clean:,} / {len(freddie_ready):,} = "
          f"{100*n_clean/len(freddie_ready):.1f}%")

    final["match_year"] = year

    # ── Column ordering ───────────────────────────────────────────────────────
    id_cols = ["loan_sequence_number", "match_year"]
    freddie_cols_in_output = [
        "credit_score", "first_payment_date", "first_time_homebuyer",
        "maturity_date", "msa", "mi_pct", "num_units",
        "occupancy_status_orig", "original_cltv", "original_dti",
        "loan_amount_r", "original_ltv", "interest_rate", "channel",
        "ppm_flag", "amortization_type", "state_code", "property_type",
        "postal_code", "loan_purpose_orig", "loan_term", "num_borrowers",
        "seller_name", "servicer_name", "super_conforming_flag",
        "pre_relief_refi_seq", "special_eligibility", "relief_refi_indicator",
        "property_valuation", "io_indicator", "mi_cancellation",
        "fp_year", "zip3",
    ]
    match_cols = ["loan_amount", "loan_amount_r", "occupancy_type", "loan_purpose"]
    demo_cols  = [c for c in HMDA_DEMO_COLS  if c in final.columns]
    extra_cols = [c for c in HMDA_EXTRA_COLS if c in final.columns]

    ordered = []
    seen    = set()
    for group in [id_cols, freddie_cols_in_output, match_cols, demo_cols, extra_cols]:
        for c in group:
            if c in final.columns and c not in seen:
                ordered.append(c); seen.add(c)
            for suffix in ["_freddie", "_hmda"]:
                c_suf = f"{c}{suffix}"
                if c_suf in final.columns and c_suf not in seen:
                    ordered.append(c_suf); seen.add(c_suf)

    remaining = [c for c in final.columns if c not in seen]
    ordered  += remaining

    final = final[ordered]

    out_path = os.path.join(output_dir, f"matched_{year}.csv")
    final.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nSaved: {out_path}  ({size_mb:.1f} MB)")
    print(f"Rows   : {len(final):,}")
    print(f"Columns: {len(final.columns)}")
    print(f"\n[A] Identifiers   : {[c for c in id_cols if c in final.columns]}")
    print(f"[B] Freddie (32)  : {[c for c in freddie_cols_in_output if c in final.columns]}")
    print(f"[D] HMDA demo     : {demo_cols}")


def combine_years(drive_root: str) -> None:
    """Combine all matched_{YEAR}.csv into one file."""
    output_dir = os.path.join(drive_root, "output")
    files = sorted(glob.glob(os.path.join(output_dir, "matched_20*.csv")))

    if not files:
        print("No matched_*.csv files found.")
        return

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    print("\nCombining... (heavy operation)")
    final_all = pd.concat(
        [pd.read_csv(f, dtype=str, low_memory=False) for f in files],
        ignore_index=True,
    )
    final_all["match_year"] = final_all["match_year"].astype(int)

    out_path = os.path.join(output_dir, "matched_all_years.csv")
    final_all.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nCombined dataset: {len(final_all):,} rows")
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")
    print("\nRows per year:")
    print(final_all.groupby("match_year").size().rename("rows").to_string())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match Freddie Mac origination data with HMDA demographics."
    )
    parser.add_argument(
        "--drive_root", required=True,
        help="Root directory (e.g. /content/drive/MyDrive/thesis_data)"
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Year to process (e.g. 2024)"
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="After matching, combine all matched_*.csv into one file"
    )
    args = parser.parse_args()

    run_match(year=args.year, drive_root=args.drive_root)

    if args.combine:
        combine_years(drive_root=args.drive_root)
