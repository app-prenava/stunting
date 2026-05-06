import zipfile
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = Path("Data/raw")
PROCESSED_DIR = Path("Data/processed")
TEMP_DIR = Path("Data/temp")

BASE_FILE = PROCESSED_DIR / "03_child_mother_maternal_features.csv"
BK_ZIP = RAW_DIR / "hh14_bk_dta.zip"
OUTPUT_FILE = PROCESSED_DIR / "04_child_mother_final_features.csv"


def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def find_dta_file(directory: Path, filename: str) -> Path:
    matches = list(directory.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found in {directory}")
    return matches[0]


def normalize_pidlink(val):
    try:
        if pd.isna(val):
            return None
        return str(int(float(val))).zfill(9)
    except (ValueError, TypeError):
        return None


def main():
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Base file not found: {BASE_FILE}")

    df_base = pd.read_csv(BASE_FILE, dtype={"household_id": str})
    rows_before = len(df_base)

    # Ensure roster file is available
    try:
        roster_path = find_dta_file(TEMP_DIR, "bk_ar1.dta")
    except FileNotFoundError:
        extract_zip(BK_ZIP, TEMP_DIR)
        roster_path = find_dta_file(TEMP_DIR, "bk_ar1.dta")

    df_roster = pd.read_stata(roster_path)
    df_roster.columns = df_roster.columns.str.lower()

    # Validate required columns exist in roster
    required_cols = ["pidlink", "ar16", "ar15c"]
    missing_cols = [c for c in required_cols if c not in df_roster.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in bk_ar1.dta: {missing_cols}")

    # Normalize pidlink for matching
    df_base["mother_pidlink_norm"] = df_base["mother_pidlink"].apply(normalize_pidlink)

    df_mother_socio = df_roster[required_cols].copy()
    df_mother_socio["pidlink_norm"] = df_mother_socio["pidlink"].apply(normalize_pidlink)
    
    # Bug fix: Filter out rows with null pidlink_norm to prevent NaN-to-NaN merge
    df_mother_socio = df_mother_socio[df_mother_socio["pidlink_norm"].notna()]
    df_mother_socio = df_mother_socio.drop_duplicates(subset=["pidlink_norm"])

    # Merge on normalized pidlink
    df_merged = df_base.merge(
        df_mother_socio[["pidlink_norm", "ar16", "ar15c"]],
        left_on="mother_pidlink_norm",
        right_on="pidlink_norm",
        how="left",
    )

    rows_after = len(df_merged)
    if rows_before != rows_after:
        raise ValueError(f"Row count changed: {rows_before} -> {rows_after}")

    # Clean up helper columns
    df_merged = df_merged.drop(columns=["mother_pidlink_norm", "pidlink_norm"], errors="ignore")

    # Rename to descriptive snake_case
    df_merged = df_merged.rename(columns={
        "ar16": "mother_education_level",
        "ar15c": "mother_employment_status",
    })

    # Validasi anak tanpa ibu
    no_mother_mask = df_merged["mother_pidlink"].isna()
    jumlah_anak_tanpa_ibu = no_mother_mask.sum()
    
    # Validasi fitur socio-economic mereka NaN
    no_mother_edu_missing = df_merged[no_mother_mask]["mother_education_level"].isna().all()
    no_mother_emp_missing = df_merged[no_mother_mask]["mother_employment_status"].isna().all()
    status_nan_valid = "Valid (All NaN)" if (no_mother_edu_missing and no_mother_emp_missing) else "Invalid (Contains Data)"

    df_merged.to_csv(OUTPUT_FILE, index=False)

    print(f"rows_before: {rows_before}")
    print(f"rows_after: {rows_after}")
    print(f"jumlah anak tanpa ibu: {jumlah_anak_tanpa_ibu}")
    print(f"validasi bahwa fitur socio-economic mereka NaN: {status_nan_valid}")
    print(f"output path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
