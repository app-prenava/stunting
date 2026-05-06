import zipfile
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = Path("Data/raw")
PROCESSED_DIR = Path("Data/processed")
TEMP_DIR = Path("Data/temp")

BASE_FILE = PROCESSED_DIR / "01_child_mother_merged.csv"
B4_ZIP = RAW_DIR / "hh14_b4_dta.zip"
OUTPUT_FILE = PROCESSED_DIR / "03_child_mother_maternal_features.csv"

def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"File ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

def find_dta_file(directory: Path, filename: str) -> Path:
    matches = list(directory.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"File {filename} not found in {directory}")
    return matches[0]

def clean_ch07_id(val):
    try:
        if pd.isna(val): 
            return None
        return float(str(val).split(':')[0].strip())
    except:
        return None

def main():
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"Base file not found: {BASE_FILE}")
        
    df_base = pd.read_csv(BASE_FILE, dtype={'household_id': str})
    rows_before = len(df_base)
    
    try:
        b4_ch1_path = find_dta_file(TEMP_DIR, "b4_ch1.dta")
    except FileNotFoundError:
        extract_zip(B4_ZIP, TEMP_DIR)
        b4_ch1_path = find_dta_file(TEMP_DIR, "b4_ch1.dta")
        
    df_ch1 = pd.read_stata(b4_ch1_path)
    df_ch1.columns = df_ch1.columns.str.lower()
    
    required_cols = ['hhid14', 'pid14', 'ch07_id', 'ch19a', 'ch20gb']
    df_ch1_selected = df_ch1[required_cols].copy()
    df_ch1_selected['hhid14'] = df_ch1_selected['hhid14'].astype(str)
    
    df_ch1_selected['child_id_match'] = df_ch1_selected['ch07_id'].apply(clean_ch07_id)
    df_ch1_selected = df_ch1_selected.drop_duplicates(subset=['hhid14', 'pid14', 'child_id_match'])
    
    df_merged = df_base.merge(
        df_ch1_selected[['hhid14', 'pid14', 'child_id_match', 'ch19a', 'ch20gb']],
        left_on=['household_id', 'mother_pid14', 'child_id'],
        right_on=['hhid14', 'pid14', 'child_id_match'],
        how='left'
    )
    
    rows_after = len(df_merged)
    
    if rows_before != rows_after:
        raise ValueError(f"Merge error: rows_before ({rows_before}) != rows_after ({rows_after})")
        
    df_merged = df_merged.drop(columns=['hhid14', 'pid14', 'child_id_match'], errors='ignore')
    
    rename_mapping = {
        'ch19a': 'anc_location',
        'ch20gb': 'delivery_insurance'
    }
    df_merged = df_merged.rename(columns=rename_mapping)
    
    missing_anc = df_merged['anc_location'].isna().sum()
    missing_insurance = df_merged['delivery_insurance'].isna().sum()
    
    df_merged.to_csv(OUTPUT_FILE, index=False)
    
    print(f"rows_before: {rows_before}")
    print(f"rows_after: {rows_after}")
    print(f"missing anc_location: {missing_anc}")
    print(f"missing delivery_insurance: {missing_insurance}")
    print(f"output path: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
