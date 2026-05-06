import pandas as pd
import zipfile
import pyreadstat
import os

print("=== STEP 4b: ENHANCED MATERNAL-HOUSEHOLD FEATURES ===")

# 1. Load Step 4 Dataset
input_path = "Data/processed/04_child_mother_final_features.csv"
print(f"Loading {input_path}...")
df_main = pd.read_csv(input_path)
print(f"Initial shape: {df_main.shape}")

# 2. Extract US (Anthropometry)
print("\nExtracting bus_us.dta for mother_height_cm...")
with zipfile.ZipFile("Data/raw/hh14_bus_dta.zip", 'r') as z:
    z.extract("bus_us.dta", "/tmp/")
df_us, _ = pyreadstat.read_dta("/tmp/bus_us.dta")
os.remove("/tmp/bus_us.dta")

# Ambil tinggi badan (us04). Batasi nilai biologis valid (100 - 250 cm) untuk membuang kode missing 999.0
df_us['mother_height_cm'] = pd.to_numeric(df_us['us04'], errors='coerce')
df_us.loc[(df_us['mother_height_cm'] < 100) | (df_us['mother_height_cm'] > 250), 'mother_height_cm'] = pd.NA

# Hanya ambil pidlink ibu dan tinggi badannya
df_us = df_us[['pidlink', 'mother_height_cm']].rename(columns={'pidlink': 'mother_pidlink'})
df_us = df_us.dropna(subset=['mother_height_cm']).drop_duplicates(subset=['mother_pidlink'])

# 3. Extract KR (Housing/Sanitation)
print("\nExtracting b2_kr.dta for improved_water & improved_sanitation...")
with zipfile.ZipFile("Data/raw/hh14_b2_dta.zip", 'r') as z:
    z.extract("b2_kr.dta", "/tmp/")
df_kr, meta_kr = pyreadstat.read_dta("/tmp/b2_kr.dta")
os.remove("/tmp/b2_kr.dta")

print("\n--- Value Counts KR13 (Drinking Water) ---")
print(df_kr['kr13'].value_counts(dropna=False).sort_index())
print("\n--- Value Counts KR20 (Toilet Facility) ---")
print(df_kr['kr20'].value_counts(dropna=False).sort_index())

# Mapping Improved Water (1=Layak, 0=Tidak Layak)
# 1:Pipe, 2:Well pump, 3:Well no pump, 4:Spring, 5:Rain, 10:Bottled -> Improved
improved_water_codes = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
df_kr['improved_water'] = df_kr['kr13'].apply(
    lambda x: 1 if x in improved_water_codes else (0 if pd.notna(x) and x not in [99.0, 95.0] else pd.NA)
)

# Mapping Improved Sanitation (1=Layak, 0=Tidak Layak)
# 1:Own toilet with septic tank -> Improved (ketat sesuai JMP agar faktor proteksi infeksi benar-benar valid)
improved_sanitation_codes = [1.0]
df_kr['improved_sanitation'] = df_kr['kr20'].apply(
    lambda x: 1 if x in improved_sanitation_codes else (0 if pd.notna(x) and x not in [99.0, 95.0] else pd.NA)
)

df_kr = df_kr[['hhid14', 'improved_water', 'improved_sanitation']].drop_duplicates(subset=['hhid14'])
df_kr = df_kr.rename(columns={'hhid14': 'household_id'})

# 4. Merge Data
print("\nMerging features into main dataset...")
# Cast merge keys to float to match the CSV parsed types and ignore leading zeros
df_main['mother_pidlink'] = df_main['mother_pidlink'].astype(float)
df_us['mother_pidlink'] = pd.to_numeric(df_us['mother_pidlink'], errors='coerce')

df_main['household_id'] = df_main['household_id'].astype(float)
df_kr['household_id'] = pd.to_numeric(df_kr['household_id'], errors='coerce')

# Left join ke mother_pidlink untuk memastikan data hanya menempel ke ibu yang tepat
df_main = df_main.merge(df_us, on='mother_pidlink', how='left')
# Left join ke household_id untuk menempelkan data fasilitas rumah
df_main = df_main.merge(df_kr, on='household_id', how='left')

print(f"Final shape after merge: {df_main.shape}")
print("\nMissing values in new features:")
print(df_main[['mother_height_cm', 'improved_water', 'improved_sanitation']].isnull().sum())

output_path = "Data/processed/04b_child_mother_enhanced.csv"
df_main.to_csv(output_path, index=False)
print(f"\nSaved enhanced dataset to: {output_path}")
