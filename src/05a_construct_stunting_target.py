import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = Path("Data/raw")
PROCESSED_DIR = Path("Data/processed")
TEMP_DIR = Path("Data/temp")

BASE_FILE = PROCESSED_DIR / "04c_child_mother_maternal_household.csv"
B5_ZIP = RAW_DIR / "hh14_b5_dta.zip"
BUS_ZIP = RAW_DIR / "hh14_bus_dta.zip"
OUTPUT_FILE = PROCESSED_DIR / "05a_child_mother_with_stunting_label_raw.csv"

# WHO Child Growth Standards 2006 - LMS parameters for Length/Height-for-Age
# Source: WHO Multicentre Growth Reference Study Group (2006)
# L = Box-Cox power, M = Median, S = Coefficient of variation
# Index: age in months (0-60), by sex (1=male, 2=female)
# For months 0-24: recumbent length; for months 25-60: standing height

WHO_LMS_BOYS = {
    0: (1, 49.8842, 0.03795), 1: (1, 54.7244, 0.03557), 2: (1, 58.4249, 0.03424),
    3: (1, 61.4292, 0.03328), 4: (1, 63.886, 0.03257), 5: (1, 65.9026, 0.03204),
    6: (1, 67.6236, 0.03165), 7: (1, 69.1645, 0.03139), 8: (1, 70.5994, 0.03124),
    9: (1, 71.9687, 0.03117), 10: (1, 73.2812, 0.03118), 11: (1, 74.5388, 0.03125),
    12: (1, 75.7488, 0.03137), 13: (1, 76.9186, 0.03154), 14: (1, 78.0497, 0.03174),
    15: (1, 79.1458, 0.03197), 16: (1, 80.2113, 0.03222), 17: (1, 81.2487, 0.03248),
    18: (1, 82.2587, 0.03274), 19: (1, 83.2418, 0.03300), 20: (1, 84.1996, 0.03326),
    21: (1, 85.1348, 0.03351), 22: (1, 86.0477, 0.03374), 23: (1, 86.941, 0.03396),
    24: (1, 87.8161, 0.03416), 25: (1, 88.0018, 0.03426), 26: (1, 88.8117, 0.03445),
    27: (1, 89.6027, 0.03463), 28: (1, 90.3754, 0.03481), 29: (1, 91.1313, 0.03499),
    30: (1, 91.8701, 0.03517), 31: (1, 92.5922, 0.03535), 32: (1, 93.2991, 0.03554),
    33: (1, 93.9919, 0.03573), 34: (1, 94.6716, 0.03592), 35: (1, 95.339, 0.03611),
    36: (1, 95.9946, 0.03630), 37: (1, 96.6395, 0.03649), 38: (1, 97.2742, 0.03668),
    39: (1, 97.8995, 0.03686), 40: (1, 98.5164, 0.03704), 41: (1, 99.1254, 0.03722),
    42: (1, 99.7271, 0.03740), 43: (1, 100.3222, 0.03758), 44: (1, 100.9113, 0.03775),
    45: (1, 101.4952, 0.03792), 46: (1, 102.0743, 0.03809), 47: (1, 102.6494, 0.03826),
    48: (1, 103.2206, 0.03842), 49: (1, 103.7886, 0.03858), 50: (1, 104.3541, 0.03874),
    51: (1, 104.9174, 0.03889), 52: (1, 105.4793, 0.03904), 53: (1, 106.0398, 0.03919),
    54: (1, 106.5997, 0.03934), 55: (1, 107.1592, 0.03949), 56: (1, 107.719, 0.03964),
    57: (1, 108.2793, 0.03979), 58: (1, 108.8405, 0.03994), 59: (1, 109.4032, 0.04009),
    60: (1, 109.9681, 0.04024),
}

WHO_LMS_GIRLS = {
    0: (1, 49.1477, 0.03790), 1: (1, 53.6872, 0.03614), 2: (1, 57.0673, 0.03568),
    3: (1, 59.8029, 0.03541), 4: (1, 62.0899, 0.03527), 5: (1, 64.0301, 0.03520),
    6: (1, 65.7311, 0.03518), 7: (1, 67.2873, 0.03524), 8: (1, 68.7498, 0.03535),
    9: (1, 70.1435, 0.03551), 10: (1, 71.4818, 0.03571), 11: (1, 72.771, 0.03594),
    12: (1, 74.015, 0.03619), 13: (1, 75.2176, 0.03647), 14: (1, 76.3817, 0.03677),
    15: (1, 77.5099, 0.03708), 16: (1, 78.6055, 0.03741), 17: (1, 79.671, 0.03774),
    18: (1, 80.7079, 0.03808), 19: (1, 81.7182, 0.03842), 20: (1, 82.7036, 0.03876),
    21: (1, 83.6654, 0.03910), 22: (1, 84.604, 0.03943), 23: (1, 85.5202, 0.03976),
    24: (1, 86.4153, 0.04008), 25: (1, 86.5904, 0.04015), 26: (1, 87.4462, 0.04046),
    27: (1, 88.2828, 0.04076), 28: (1, 89.1004, 0.04107), 29: (1, 89.8997, 0.04137),
    30: (1, 90.6817, 0.04167), 31: (1, 91.4471, 0.04197), 32: (1, 92.1968, 0.04226),
    33: (1, 92.9316, 0.04255), 34: (1, 93.6523, 0.04284), 35: (1, 94.3598, 0.04312),
    36: (1, 95.0551, 0.04340), 37: (1, 95.739, 0.04367), 38: (1, 96.4122, 0.04395),
    39: (1, 97.0756, 0.04422), 40: (1, 97.7302, 0.04448), 41: (1, 98.3768, 0.04475),
    42: (1, 99.0162, 0.04501), 43: (1, 99.6492, 0.04527), 44: (1, 100.2765, 0.04553),
    45: (1, 100.8989, 0.04578), 46: (1, 101.5169, 0.04603), 47: (1, 102.1314, 0.04628),
    48: (1, 102.7431, 0.04653), 49: (1, 103.3524, 0.04677), 50: (1, 103.9602, 0.04701),
    51: (1, 104.567, 0.04725), 52: (1, 105.1733, 0.04749), 53: (1, 105.7797, 0.04773),
    54: (1, 106.3868, 0.04797), 55: (1, 106.9953, 0.04821), 56: (1, 107.6055, 0.04845),
    57: (1, 108.2183, 0.04869), 58: (1, 108.8342, 0.04893), 59: (1, 109.4535, 0.04916),
    60: (1, 110.0764, 0.04940),
}


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


def parse_ifls_numeric(val):
    try:
        if pd.isna(val):
            return None
        s = str(val).split(':')[0].strip()
        v = int(float(s))
        return v if v < 98 else None
    except (ValueError, TypeError):
        return None


def parse_ifls_year(val):
    try:
        if pd.isna(val):
            return None
        s = str(val).split(':')[0].strip()
        v = int(float(s))
        return v if v < 9000 else None
    except (ValueError, TypeError):
        return None


def parse_sex(val):
    """1:Male -> 1, 3:Female -> 2 (WHO convention)"""
    try:
        s = str(val).split(':')[0].strip()
        code = int(float(s))
        if code == 1:
            return 1
        if code == 3:
            return 2
        return None
    except (ValueError, TypeError):
        return None


def compute_haz(height_cm, age_months, sex):
    """
    Compute Height-for-Age Z-score using WHO 2006 LMS method.
    Formula: Z = ((height/M)^L - 1) / (L * S)
    When L=1 (which is true for all WHO HAZ): Z = (height - M) / (M * S)
    """
    if pd.isna(height_cm) or pd.isna(age_months) or pd.isna(sex):
        return np.nan

    age_months = int(age_months)
    if age_months < 0 or age_months > 60:
        return np.nan

    lms_table = WHO_LMS_BOYS if sex == 1 else WHO_LMS_GIRLS
    if age_months not in lms_table:
        return np.nan

    L, M, S = lms_table[age_months]

    if L == 1:
        z = (height_cm - M) / (M * S)
    else:
        z = (((height_cm / M) ** L) - 1) / (L * S)

    return round(z, 4)


def main():
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Base file not found: {BASE_FILE}")

    df_base = pd.read_csv(BASE_FILE, dtype={"household_id": str})
    rows_before = len(df_base)

    # Ensure source files are extracted
    for zip_path in [B5_ZIP, BUS_ZIP]:
        try:
            extract_zip(zip_path, TEMP_DIR)
        except FileNotFoundError:
            raise

    # Load child DOB from b5_cov
    b5_cov_path = find_dta_file(TEMP_DIR, "b5_cov.dta")
    df_b5 = pd.read_stata(b5_cov_path)
    df_b5.columns = df_b5.columns.str.lower()

    # Load interview date from b5_time
    b5_time_path = find_dta_file(TEMP_DIR, "b5_time.dta")
    df_time = pd.read_stata(b5_time_path)
    df_time.columns = df_time.columns.str.lower()

    # Load anthropometry from bus_us
    bus_us_path = find_dta_file(TEMP_DIR, "bus_us.dta")
    df_us = pd.read_stata(bus_us_path)
    df_us.columns = df_us.columns.str.lower()

    # Normalize pidlinks
    df_base["child_pidlink_norm"] = df_base["child_unique_id"].apply(
        lambda x: str(int(x)).zfill(9) if pd.notna(x) else None
    )
    df_b5["pidlink_norm"] = df_b5["pidlink"].apply(normalize_pidlink)
    df_time["pidlink_norm"] = df_time["pidlink"].apply(normalize_pidlink)
    df_us["pidlink_norm"] = df_us["pidlink"].apply(normalize_pidlink)

    # Parse DOB
    df_b5["birth_month"] = df_b5["dob_mth"].apply(parse_ifls_numeric)
    df_b5["birth_year"] = df_b5["dob_yr"].apply(parse_ifls_year)

    # Get interview date (deduplicate per child)
    df_time_dedup = df_time.drop_duplicates(subset=["pidlink_norm"])

    # Build age-in-months table
    df_age = df_b5[["pidlink_norm", "birth_month", "birth_year"]].merge(
        df_time_dedup[["pidlink_norm", "ivwmth", "ivwyr"]],
        on="pidlink_norm",
        how="inner",
    )
    df_age["age_in_months"] = (
        (df_age["ivwyr"] - df_age["birth_year"]) * 12
        + (df_age["ivwmth"] - df_age["birth_month"])
    )
    df_age = df_age[["pidlink_norm", "age_in_months"]].drop_duplicates(
        subset=["pidlink_norm"]
    )

    # Build height table
    df_height = df_us[["pidlink_norm", "us04", "us05"]].copy()
    df_height = df_height.rename(columns={"us04": "height_cm", "us05": "measure_position"})
    df_height = df_height.drop_duplicates(subset=["pidlink_norm"])

    # Parse sex from base dataset
    df_base["sex_who"] = df_base["child_gender"].apply(parse_sex)

    # Merge age_in_months
    df_merged = df_base.merge(
        df_age, left_on="child_pidlink_norm", right_on="pidlink_norm", how="left"
    )
    df_merged = df_merged.drop(columns=["pidlink_norm"], errors="ignore")

    # Merge height
    df_merged = df_merged.merge(
        df_height, left_on="child_pidlink_norm", right_on="pidlink_norm", how="left"
    )
    df_merged = df_merged.drop(columns=["pidlink_norm"], errors="ignore")

    rows_after = len(df_merged)
    if rows_before != rows_after:
        raise ValueError(f"Row count changed: {rows_before} -> {rows_after}")

    # Adjust height for measurement position:
    # WHO standard uses recumbent length for 0-24 months and standing height for 24-60 months.
    # If child was measured standing but age < 24 months, add 0.7 cm.
    # If child was measured lying but age >= 24 months, subtract 0.7 cm.
    def adjust_height(row):
        h = row["height_cm"]
        age = row["age_in_months"]
        pos = str(row["measure_position"])
        if pd.isna(h) or pd.isna(age):
            return h
        if age < 24 and "Standing" in pos:
            return h + 0.7
        if age >= 24 and "Lying" in pos:
            return h - 0.7
        return h

    df_merged["height_adjusted"] = df_merged.apply(adjust_height, axis=1)

    # Compute HAZ
    df_merged["haz_score"] = df_merged.apply(
        lambda r: compute_haz(r["height_adjusted"], r["age_in_months"], r["sex_who"]),
        axis=1,
    )

    # Label stunting: HAZ < -2 SD
    df_merged["is_stunted"] = np.where(
        df_merged["haz_score"].notna(),
        (df_merged["haz_score"] < -2).astype(int),
        np.nan,
    )

    # Clean up helper columns
    df_merged = df_merged.drop(
        columns=["child_pidlink_norm", "sex_who", "measure_position", "height_adjusted"],
        errors="ignore",
    )

    # Summary
    total = len(df_merged)
    has_haz = df_merged["haz_score"].notna().sum()
    missing_haz = df_merged["haz_score"].isna().sum()
    stunted = (df_merged["is_stunted"] == 1).sum()
    normal = (df_merged["is_stunted"] == 0).sum()
    prevalence = (stunted / has_haz * 100) if has_haz > 0 else 0

    df_merged.to_csv(OUTPUT_FILE, index=False)

    print(f"rows_before: {rows_before}")
    print(f"rows_after: {rows_after}")
    print(f"total children: {total}")
    print(f"haz_score available: {has_haz}")
    print(f"haz_score missing: {missing_haz}")
    print(f"stunted (HAZ < -2): {stunted}")
    print(f"normal (HAZ >= -2): {normal}")
    print(f"stunting prevalence: {prevalence:.1f}%")
    print(f"haz_score mean: {df_merged['haz_score'].mean():.4f}")
    print(f"haz_score std: {df_merged['haz_score'].std():.4f}")
    print(f"output path: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
