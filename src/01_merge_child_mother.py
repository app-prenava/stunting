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

ZIP_FILES = {
    "b5": RAW_DIR / "hh14_b5_dta.zip",
    "bk": RAW_DIR / "hh14_bk_dta.zip",
    "b4": RAW_DIR / "hh14_b4_dta.zip",
}


def extract_zip(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(TEMP_DIR)


def find_file(name):
    matches = list(TEMP_DIR.rglob(name))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def load_df(path):
    df = pd.read_stata(path)
    df.columns = df.columns.str.lower()
    return df


def main():
    for z in ZIP_FILES.values():
        extract_zip(z)

    b5 = load_df(find_file("b5_cov.dta"))
    bk = load_df(find_file("bk_ar1.dta"))
    b4 = load_df(find_file("b4_cov.dta"))

    child = b5[(b5["age"] >= 0) & (b5["age"] <= 4)].copy()
    child = child[["hhid14", "pid14", "pidlink", "age", "sex"]]

    roster = bk[["hhid14", "pid14", "ar11"]].copy()

    child = child.merge(roster, on=["hhid14", "pid14"], how="left")

    mother = b4[["hhid14", "pid14", "pidlink", "age"]].copy()
    mother = mother.rename(columns={
        "pid14": "mother_pid14",
        "pidlink": "mother_pidlink",
        "age": "mother_age"
    })

    df = child.merge(
        mother,
        left_on=["hhid14", "ar11"],
        right_on=["hhid14", "mother_pid14"],
        how="left"
    )

    df = df.rename(columns={
        "hhid14": "household_id",
        "pid14": "child_id",
        "pidlink": "child_unique_id",
        "age": "child_age",
        "sex": "child_gender"
    })

    total = len(df)
    matched = df["mother_pidlink"].notna().sum()
    unmatched = df["mother_pidlink"].isna().sum()

    print("total:", total)
    print("matched:", matched)
    print("unmatched:", unmatched)

    out = PROCESSED_DIR / "01_child_mother_merged.csv"
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()