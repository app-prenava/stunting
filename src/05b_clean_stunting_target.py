import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

PROCESSED_DIR = Path("Data/processed")
INPUT_FILE = PROCESSED_DIR / "05a_child_mother_with_stunting_label_raw.csv"
OUTPUT_FILE = PROCESSED_DIR / "05b_child_mother_stunting_cleaned.csv"

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype={"household_id": str})
    rows_before_cleaning = len(df)
    print(f"rows_before_cleaning: {rows_before_cleaning}")

    # 1. Validasi Missing Target
    missing_haz = df["haz_score"].isna().sum()
    missing_target = df["is_stunted"].isna().sum()
    print(f"missing haz_score: {missing_haz}")
    print(f"missing is_stunted: {missing_target}")

    # Drop missing target
    df_cleaned = df.dropna(subset=["haz_score", "is_stunted"])
    rows_after_drop_missing = len(df_cleaned)
    print(f"rows_after_drop_missing: {rows_after_drop_missing}")

    # 2. Validasi dan Cleaning Outlier (WHO: -6 to +6)
    outliers_mask = (df_cleaned["haz_score"] < -6) | (df_cleaned["haz_score"] > 6)
    num_outliers = outliers_mask.sum()
    print(f"num_outliers (HAZ < -6 or HAZ > +6): {num_outliers}")

    # Drop outliers
    df_cleaned = df_cleaned[~outliers_mask]
    rows_after_cleaning = len(df_cleaned)
    print(f"rows_after_cleaning: {rows_after_cleaning}")

    # 3. Statistik Dataset Final
    total_children = len(df_cleaned)
    stunted = (df_cleaned["is_stunted"] == 1).sum()
    normal = (df_cleaned["is_stunted"] == 0).sum()
    prevalence = (stunted / total_children * 100) if total_children > 0 else 0
    mean_haz = df_cleaned["haz_score"].mean()
    std_haz = df_cleaned["haz_score"].std()

    print(f"total children: {total_children}")
    print(f"stunted: {stunted}")
    print(f"normal: {normal}")
    print(f"prevalence: {prevalence:.1f}%")
    print(f"mean haz_score: {mean_haz:.4f}")
    print(f"std haz_score: {std_haz:.4f}")

    # 4. Validasi Logika Epidemiologi
    print("\n--- Epidemiologi: Prevalensi per Usia ---")
    for age in sorted(df_cleaned["child_age"].unique()):
        subset = df_cleaned[df_cleaned["child_age"] == age]
        s = (subset["is_stunted"] == 1).sum()
        t = len(subset)
        print(f"Usia {int(age)}: {s}/{t} ({s/t*100:.1f}%)")

    print("\n--- Epidemiologi: Prevalensi per Gender ---")
    for gender in df_cleaned["child_gender"].unique():
        subset = df_cleaned[df_cleaned["child_gender"] == gender]
        s = (subset["is_stunted"] == 1).sum()
        t = len(subset)
        print(f"Gender {gender}: {s}/{t} ({s/t*100:.1f}%)")

    # Simpan output
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"\noutput path: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
