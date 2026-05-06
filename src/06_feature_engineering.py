import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path("Data/processed")
INPUT_FILE = PROCESSED_DIR / "05b_child_mother_stunting_cleaned.csv"
OUTPUT_FILE = PROCESSED_DIR / "dataset_final.csv"

IFLS_MISSING_PATTERNS = ["98:Don't know", "99:Missing", "9:Missing", "98:DK"]

EDUCATION_ORDINAL = {
    "1:Unschooled": 0,
    "2:Grade school": 1,
    "72:Madrasah Ibtidaiyah": 1,
    "11:Education A": 1,
    "12:Education B": 1,
    "15:Education C": 1,
    "3:General jr. high": 2,
    "4:Vocational jr. high": 2,
    "73:Madrasah Tsanawiyah": 2,
    "14:Moslem School (Pesantren)": 2,
    "17:School for the disabled": 2,
    "5:General sr. high (SLA)": 3,
    "6:Vocational sr. high (SMK)": 3,
    "74:Madrasah Aliyah": 3,
    "13:Open University": 4,
    "60:Diploma (D1, D2, D3)": 4,
    "61:University S1": 5,
    "62:University S2": 5,
    "63:University S3": 5,
    "95:Other": 1,
}

WORKING_CATEGORIES = ["1:Work/helping to get income"]

ANC_GROUP_MAP = {
    "1:Public hospital": "hospital",
    "2:Private hospital": "hospital",
    "3:Delivery Hospital": "hospital",
    "4:Community health center": "clinic_midwife",
    "5:Village Delivery Post": "clinic_midwife",
    "6:Clinic/office of physician": "clinic_midwife",
    "7:Clinic/office of midwife": "clinic_midwife",
    "8:Office/house of trad. midwife": "traditional_other",
    "95:Other": "traditional_other",
}

ID_COLUMNS = [
    "household_id", "child_id", "child_unique_id", "ar11",
    "mother_pid14", "mother_pidlink",
]

RAW_TEXT_COLUMNS = [
    "anc_location", "delivery_insurance",
    "child_age", "mother_age",
]

POST_ENCODING_DROP = ["haz_score", "height_cm", "age_in_months"]


def clean_ifls_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        mask = df[col].isin(IFLS_MISSING_PATTERNS)
        mask |= df[col].str.contains("Don't know", case=False, na=False)
        mask |= df[col].str.match(r"^99:", na=False)
        df.loc[mask, col] = np.nan
    return df


def encode_gender(series: pd.Series) -> pd.Series:
    return series.map({"1:Male": 1, "3:Female": 0})


def encode_education(series: pd.Series) -> pd.Series:
    return series.map(EDUCATION_ORDINAL)


def encode_employment(series: pd.Series) -> pd.Series:
    return series.isin(WORKING_CATEGORIES).astype(int).where(series.notna(), np.nan)


def encode_anc(df: pd.DataFrame) -> pd.DataFrame:
    df["anc_group"] = df["anc_location"].map(ANC_GROUP_MAP)
    df.loc[df["anc_group"].isna(), "anc_group"] = "unknown"
    dummies = pd.get_dummies(df["anc_group"], prefix="anc", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=["anc_group"])
    return df


def encode_insurance(series: pd.Series) -> pd.Series:
    return series.notna().astype(int)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype={"household_id": str})
    rows_input = len(df)

    # 1. Clean IFLS missing patterns (NOT treating 95:Other as missing)
    df = clean_ifls_missing(df)

    # 2. Derived features
    df["mother_age_at_birth"] = df["mother_age"] - (df["age_in_months"] / 12)
    df["is_teenage_mother"] = (df["mother_age_at_birth"] < 20).astype(int).where(
        df["mother_age_at_birth"].notna(), np.nan
    )
    df["is_high_risk_mother_age"] = (
        (df["mother_age_at_birth"] < 18) | (df["mother_age_at_birth"] > 35)
    ).astype(int).where(df["mother_age_at_birth"].notna(), np.nan)

    # 3. Encode features
    df["child_gender"] = encode_gender(df["child_gender"])
    df["mother_education_level"] = encode_education(df["mother_education_level"])
    df["mother_employment_status"] = encode_employment(df["mother_employment_status"])
    df["has_delivery_insurance"] = encode_insurance(df["delivery_insurance"])
    df = encode_anc(df)

    # 4. Drop ID and raw text columns
    drop_cols = ID_COLUMNS + RAW_TEXT_COLUMNS + POST_ENCODING_DROP + ["delivery_insurance", "anc_location", "quality_floor", "quality_wall", "quality_roof"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 5. Drop rows with missing critical features
    critical = [
        "mother_age_at_birth", "mother_education_level", "child_gender",
        "mother_height_cm", "improved_water", "improved_sanitation",
        "home_ownership", "has_electricity", "has_refrigerator", "has_tv"
    ]
    rows_before_drop = len(df)
    df = df.dropna(subset=critical)
    rows_after_drop = len(df)

    # 6. Handle missing mother_employment_status: impute with 0 (not working)
    emp_missing_count = df["mother_employment_status"].isna().sum()
    df["mother_employment_status"] = df["mother_employment_status"].fillna(0)

    # 7. Cast all columns to proper integer types where possible
    df["is_stunted"] = df["is_stunted"].astype(int)
    df["mother_education_level"] = df["mother_education_level"].astype(int)
    df["mother_employment_status"] = df["mother_employment_status"].astype(int)
    df["is_teenage_mother"] = df["is_teenage_mother"].astype(int)
    df["is_high_risk_mother_age"] = df["is_high_risk_mother_age"].astype(int)

    # 8. Validate mother_age_at_birth
    invalid_age = (df["mother_age_at_birth"] < 0) | (df["mother_age_at_birth"] > 60)
    num_invalid_age = invalid_age.sum()

    df.to_csv(OUTPUT_FILE, index=False)

    # === VALIDATION OUTPUT ===
    print(f"rows_input: {rows_input}")
    print(f"rows_dropped_critical_missing: {rows_before_drop - rows_after_drop}")
    print(f"mother_employment_status imputed: {emp_missing_count}")
    print(f"mother_age_at_birth invalid (neg or >60): {num_invalid_age}")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print()

    # Banned columns check
    banned = ["household_id", "child_id", "child_unique_id", "mother_pid14",
              "mother_pidlink", "haz_score", "height_cm", "age_in_months"]
    leaked = [c for c in banned if c in df.columns]
    print(f"banned columns present: {leaked if leaked else 'NONE (clean)'}")
    print()

    # Missing per column
    print("missing per column:")
    total_missing = 0
    for col in df.columns:
        m = df[col].isna().sum()
        total_missing += m
        print(f"  {col}: {m}")
    print(f"  TOTAL: {total_missing}")
    print()

    # Dtypes check
    print("dtypes:")
    all_numeric = True
    for col in df.columns:
        dtype = df[col].dtype
        is_num = pd.api.types.is_numeric_dtype(dtype)
        if not is_num:
            all_numeric = False
        print(f"  {col}: {dtype} {'(numeric)' if is_num else '(NOT NUMERIC)'}")
    print(f"  all numeric: {all_numeric}")
    print()

    # Target validation
    stunted = (df["is_stunted"] == 1).sum()
    normal = (df["is_stunted"] == 0).sum()
    print(f"is_stunted distribution:")
    print(f"  stunted: {stunted} ({stunted / len(df) * 100:.1f}%)")
    print(f"  normal: {normal} ({normal / len(df) * 100:.1f}%)")
    print(f"  missing: {df['is_stunted'].isna().sum()}")
    print()

    # mother_age_at_birth stats
    mab = df["mother_age_at_birth"]
    print(f"mother_age_at_birth:")
    print(f"  mean: {mab.mean():.2f}")
    print(f"  std: {mab.std():.2f}")
    print(f"  min: {mab.min():.2f}")
    print(f"  max: {mab.max():.2f}")
    print()

    print("sample (first 5 rows):")
    print(df.head().to_string())
    print()
    print(f"output path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
