import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

RAW_DIR = Path('Data/raw')
PROCESSED_DIR = Path('Data/processed')

# Load Base Dataset
df_base = pd.read_csv(PROCESSED_DIR / '05b_child_mother_stunting_cleaned.csv')

def format_hhid14(x):
    return f"{int(x):07d}" if pd.notnull(x) else ""

def format_pidlink(x):
    return f"{int(x):09d}" if pd.notnull(x) else ""

df_base['hhid14'] = df_base['household_id'].apply(format_hhid14)
df_base['mother_pidlink_str'] = df_base['mother_pidlink'].apply(format_pidlink)
df_base['child_pid14'] = df_base['child_id'].astype(float)

# ==========================================
# 1. MATERNAL LEVEL (b4_br, b4_kw3, bus_us)
# ==========================================
print("Extracting Maternal Level features...")
# a) Fertility & Miscarriages
with zipfile.ZipFile(RAW_DIR / 'hh14_b4_dta.zip', 'r') as z:
    with z.open('b4_br.dta') as f:
        df_br = pd.read_stata(f)
        df_br = df_br[['pidlink', 'br14', 'br15']]
        df_br['br14'] = pd.to_numeric(df_br['br14'], errors='coerce').fillna(0) # miscarriages
        df_br['br15'] = pd.to_numeric(df_br['br15'], errors='coerce') # parity
        df_br.rename(columns={'br14': 'maternal_miscarriages', 'br15': 'maternal_parity'}, inplace=True)

# b) Marriage Age
with zipfile.ZipFile(RAW_DIR / 'hh14_b4_dta.zip', 'r') as z:
    with z.open('b4_kw3.dta') as f:
        df_kw3 = pd.read_stata(f)
        df_kw3 = df_kw3[df_kw3['kwn_num'] == 1] # First marriage
        df_kw3['kw11'] = pd.to_numeric(df_kw3['kw11'], errors='coerce')
        df_kw3 = df_kw3[['pidlink', 'kw11']].rename(columns={'kw11': 'maternal_marriage_age'})

# c) Mother's BMI and Hemoglobin
with zipfile.ZipFile(RAW_DIR / 'hh14_bus_dta.zip', 'r') as z:
    with z.open('bus_us.dta') as f:
        df_bus = pd.read_stata(f)
        df_bus['us06'] = pd.to_numeric(df_bus['us06'], errors='coerce') # weight
        df_bus['us04'] = pd.to_numeric(df_bus['us04'], errors='coerce') # height
        df_bus['us13'] = pd.to_numeric(df_bus['us13'], errors='coerce') # hb
        df_bus['maternal_bmi'] = df_bus['us06'] / ((df_bus['us04']/100) ** 2)
        df_bus.loc[(df_bus['maternal_bmi'] > 50) | (df_bus['maternal_bmi'] < 10), 'maternal_bmi'] = np.nan
        df_bus = df_bus[['pidlink', 'maternal_bmi', 'us13']].rename(columns={'us13': 'maternal_hemoglobin'})

# Merge Maternal to Base
df_base = df_base.merge(df_br, left_on='mother_pidlink_str', right_on='pidlink', how='left').drop('pidlink', axis=1)
df_base = df_base.merge(df_kw3, left_on='mother_pidlink_str', right_on='pidlink', how='left').drop('pidlink', axis=1)
df_base = df_base.merge(df_bus, left_on='mother_pidlink_str', right_on='pidlink', how='left').drop('pidlink', axis=1)


# ==========================================
# 2. CHILD / PREGNANCY LEVEL (b4_ch1, b5_maa2)
# ==========================================
print("Extracting Child Level features...")
# a) Birth Weight, ASI, ANC visits
with zipfile.ZipFile(RAW_DIR / 'hh14_b4_dta.zip', 'r') as z:
    with z.open('b4_ch1.dta') as f:
        df_ch1 = pd.read_stata(f)
        df_ch1['ch07_id'] = pd.to_numeric(df_ch1['ch07_id'], errors='coerce')
        df_ch1['ch24'] = pd.to_numeric(df_ch1['ch24'], errors='coerce') # Birth weight
        # ch24a: Breastfed (1: Yes, 3: No)
        df_ch1['child_breastfed'] = df_ch1['ch24a'].apply(lambda x: 1 if str(x).startswith('1') else (0 if str(x).startswith('3') else np.nan))
        
        for col in ['ch16a', 'ch16b', 'ch16c']:
            df_ch1[col] = pd.to_numeric(df_ch1[col], errors='coerce').fillna(0)
        df_ch1['maternal_anc_visits'] = df_ch1['ch16a'] + df_ch1['ch16b'] + df_ch1['ch16c']
        df_ch1.loc[df_ch1['maternal_anc_visits'] > 30, 'maternal_anc_visits'] = np.nan # Outliers
        
        df_ch1 = df_ch1[['hhid14', 'ch07_id', 'ch24', 'child_breastfed', 'maternal_anc_visits']].rename(columns={'ch24': 'child_birth_weight'})
        # Aggregate to avoid duplicates
        df_ch1 = df_ch1.groupby(['hhid14', 'ch07_id']).first().reset_index()

df_base = df_base.merge(df_ch1, left_on=['hhid14', 'child_pid14'], right_on=['hhid14', 'ch07_id'], how='left').drop('ch07_id', axis=1)

# b) Morbidity (Diarrhea/ISPA)
with zipfile.ZipFile(RAW_DIR / 'hh14_b5_dta.zip', 'r') as z:
    with z.open('b5_maa2.dta') as f:
        df_maa2 = pd.read_stata(f)
        # maa01: 1=Yes, 3=No. Did child have symptom in last 4 weeks.
        df_maa2['child_had_diarrhea'] = ((df_maa2['maatype'].astype(str) == 'BA') & (df_maa2['maa01'].astype(str).str.startswith('1'))).astype(int)
        df_maa2['child_had_ari'] = ((df_maa2['maatype'].astype(str) == 'CA') & (df_maa2['maa01'].astype(str).str.startswith('1'))).astype(int)
        df_morb = df_maa2.groupby('pidlink')[['child_had_diarrhea', 'child_had_ari']].max().reset_index()
        # Merge by pidlink
        df_base['child_pidlink_str'] = df_base['child_unique_id'].apply(format_pidlink)
        df_base = df_base.merge(df_morb, left_on='child_pidlink_str', right_on='pidlink', how='left').drop(['pidlink', 'child_pidlink_str'], axis=1)


# ==========================================
# 3. HOUSEHOLD LEVEL (b1_ks1)
# ==========================================
print("Extracting Household Level features...")
# a) Expenditure
with zipfile.ZipFile(RAW_DIR / 'hh14_b1_dta.zip', 'r') as z:
    with z.open('b1_ks1.dta') as f:
        df_ks1 = pd.read_stata(f)
        # ks02: total expenditure for food past week
        df_ks1['ks02'] = pd.to_numeric(df_ks1['ks02'], errors='coerce')
        df_ks1 = df_ks1[['hhid14', 'ks02']].groupby('hhid14').sum().reset_index()
        df_ks1.rename(columns={'ks02': 'hh_food_expenditure_weekly'}, inplace=True)

df_base = df_base.merge(df_ks1, on='hhid14', how='left')


# Clean up and export
drop_cols = ['hhid14', 'mother_pidlink_str', 'child_pid14']
df_final = df_base.drop(columns=drop_cols, errors='ignore')

print(f"\nFinal dataset shape: {df_final.shape}")
print("New columns added:")
new_cols = ['maternal_miscarriages', 'maternal_parity', 'maternal_marriage_age', 'maternal_bmi', 'maternal_hemoglobin', 'child_birth_weight', 'child_breastfed', 'maternal_anc_visits', 'child_had_diarrhea', 'child_had_ari', 'hh_food_expenditure_weekly']
for c in new_cols:
    if c in df_final.columns:
        print(f"  - {c}: {df_final[c].notnull().sum()} non-null values")

df_final.to_csv(PROCESSED_DIR / '05c_child_mother_augmented.csv', index=False)
print("Saved to 05c_child_mother_augmented.csv!")
