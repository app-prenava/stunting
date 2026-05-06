import pandas as pd
import zipfile
import pyreadstat
import os

print("=== STEP 4c: MATERNAL-HOUSEHOLD WEALTH FEATURES ===")

# 1. Load Step 4b Dataset
input_path = "Data/processed/04b_child_mother_enhanced.csv"
print(f"Loading {input_path}...")
df_main = pd.read_csv(input_path)
print(f"Initial shape: {df_main.shape}")

# 2. Extract KR (Household Characteristics)
print("\nExtracting b2_kr.dta for ownership, electricity, and assets...")
with zipfile.ZipFile("Data/raw/hh14_b2_dta.zip", 'r') as z:
    z.extract("b2_kr.dta", "/tmp/")
df_kr, _ = pyreadstat.read_dta("/tmp/b2_kr.dta")
os.remove("/tmp/b2_kr.dta")

df_kr['household_id'] = pd.to_numeric(df_kr['hhid14'], errors='coerce')
df_kr = df_kr.drop_duplicates(subset=['household_id'])

# Mapping
# Home ownership (kr03): 1=Self owned -> 1, others -> 0
df_kr['home_ownership'] = df_kr['kr03'].apply(
    lambda x: 1 if x == 1.0 else (0 if pd.notna(x) and x not in [98.0, 99.0] else pd.NA)
)

# Electricity (kr11): 1=Yes -> 1, 3=No -> 0
df_kr['has_electricity'] = df_kr['kr11'].apply(
    lambda x: 1 if x == 1.0 else (0 if x == 3.0 else pd.NA)
)

# Asset: Refrigerator (kr23) & TV (kr24a) -> Combine as a single score or separate
# Let's keep separate binaries as user said "engga nanti aja" to combination
df_kr['has_refrigerator'] = df_kr['kr23'].apply(
    lambda x: 1 if x == 1.0 else (0 if x in [3.0, 6.0] else pd.NA)
)

df_kr['has_tv'] = df_kr['kr24a'].apply(
    lambda x: 1 if x == 1.0 else (0 if x == 3.0 else pd.NA)
)

df_kr_subset = df_kr[['household_id', 'home_ownership', 'has_electricity', 'has_refrigerator', 'has_tv']]

# 3. Extract KRK (Housing physical quality from BK)
print("\nExtracting bk_krk.dta for housing physical quality...")
with zipfile.ZipFile("Data/raw/hh14_bk_dta.zip", 'r') as z:
    z.extract("bk_krk.dta", "/tmp/")
df_krk, _ = pyreadstat.read_dta("/tmp/bk_krk.dta")
os.remove("/tmp/bk_krk.dta")

df_krk['household_id'] = pd.to_numeric(df_krk['hhid14'], errors='coerce')
df_krk = df_krk.drop_duplicates(subset=['household_id'])

# Mapping floor (krk08): 1,2,3 -> 1 (quality), 4,5,6 -> 0
df_krk['quality_floor'] = df_krk['krk08'].apply(
    lambda x: 1 if x in [1.0, 2.0, 3.0] else (0 if x in [4.0, 5.0, 6.0] else pd.NA)
)

# Mapping wall (krk09): 1 -> 1 (quality), 2,3 -> 0
df_krk['quality_wall'] = df_krk['krk09'].apply(
    lambda x: 1 if x == 1.0 else (0 if x in [2.0, 3.0] else pd.NA)
)

# Mapping roof (krk10): 1,3,4,5 -> 1 (quality), 2,6 -> 0
df_krk['quality_roof'] = df_krk['krk10'].apply(
    lambda x: 1 if x in [1.0, 3.0, 4.0, 5.0] else (0 if x in [2.0, 6.0] else pd.NA)
)

df_krk_subset = df_krk[['household_id', 'quality_floor', 'quality_wall', 'quality_roof']]

# 4. Merge Data
print("\nMerging features into main dataset...")
df_main['household_id'] = df_main['household_id'].astype(float)
df_main = df_main.merge(df_kr_subset, on='household_id', how='left')
df_main = df_main.merge(df_krk_subset, on='household_id', how='left')

print(f"Final shape after merge: {df_main.shape}")
new_cols = ['home_ownership', 'has_electricity', 'has_refrigerator', 'has_tv', 
            'quality_floor', 'quality_wall', 'quality_roof']
print("\nMissing values in new features:")
print(df_main[new_cols].isnull().sum())

# 5. Inventory
inventory_rows = []
for col in new_cols:
    src = "b2_kr" if col in ['home_ownership', 'has_electricity', 'has_refrigerator', 'has_tv'] else "bk_krk"
    missing = df_main[col].isna().sum()
    pct = missing / len(df_main) * 100
    vc = df_main[col].value_counts().to_dict()
    inventory_rows.append({
        "source_file": src,
        "column_name": col,
        "dtype": str(df_main[col].dtype),
        "missing_count": missing,
        "missing_percentage": round(pct, 2),
        "sample_values": str(vc),
        "candidate_feature": col,
        "status": "Accepted",
        "notes": "Kept as separate binary feature"
    })

pd.DataFrame(inventory_rows).to_csv("Data/processed/04c_maternal_household_wealth_inventory.csv", index=False)

output_path = "Data/processed/04c_child_mother_maternal_household.csv"
df_main.to_csv(output_path, index=False)
print(f"\nSaved enhanced dataset to: {output_path}")
