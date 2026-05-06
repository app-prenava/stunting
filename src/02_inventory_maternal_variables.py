import zipfile
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

RAW_DIR = Path("Data/raw")
TEMP_DIR = Path("Data/temp")
PROCESSED_DIR = Path("Data/processed")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

B4_ZIP = RAW_DIR / "hh14_b4_dta.zip"


def extract_zip(zip_path, extract_to):
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def get_dta_paths():
    names = []
    with zipfile.ZipFile(B4_ZIP, "r") as z:
        for f in z.namelist():
            if f.endswith(".dta"):
                names.append(Path(f).name)

    paths = []
    for name in names:
        found = list(TEMP_DIR.rglob(name))
        if found:
            paths.append(found[0])

    return sorted(list(set(paths)))


def build_inventory(paths):
    rows = []

    for p in paths:
        try:
            df = pd.read_stata(p)
            df.columns = df.columns.str.lower()
            n = len(df)

            for c in df.columns:
                m = df[c].isna().sum()
                pct = (m / n * 100) if n > 0 else 0
                vals = df[c].dropna().unique()[:5]
                vals = " | ".join([str(v) for v in vals])

                rows.append({
                    "source_file": p.name,
                    "column_name": c,
                    "dtype": str(df[c].dtype),
                    "missing_count": m,
                    "missing_percentage": round(pct, 2),
                    "sample_values": vals,
                    "status": "needs_codebook_validation"
                })

        except Exception:
            continue

    return pd.DataFrame(rows)


def main():
    extract_zip(B4_ZIP, TEMP_DIR)
    paths = get_dta_paths()
    df_inventory = build_inventory(paths)

    out = PROCESSED_DIR / "02_maternal_variable_inventory.csv"
    df_inventory.to_csv(out, index=False)

    print("saved:", out)


if __name__ == "__main__":
    main()