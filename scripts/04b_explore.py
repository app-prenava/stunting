import pandas as pd
import zipfile
import pyreadstat
import os

results = []

def extract_and_explore(zip_path, file_name, keywords):
    with zipfile.ZipFile(zip_path, 'r') as z:
        tmp_path = f"/tmp/{file_name}"
        with open(tmp_path, "wb") as f:
            f.write(z.read(file_name))
        
        df, meta = pyreadstat.read_dta(tmp_path)
        
        for col, label in meta.column_names_to_labels.items():
            label_lower = str(label).lower()
            if any(kw in label_lower for kw in keywords):
                missing_pct = df[col].isnull().mean() * 100
                samples = df[col].dropna().unique()[:3].tolist()
                
                results.append({
                    "Source_File": file_name,
                    "Column_IFLS": col,
                    "Label": label,
                    "Missing_Percentage": f"{missing_pct:.2f}%",
                    "Sample_Values": str(samples)
                })
        
        os.remove(tmp_path)

# Extract height from US
extract_and_explore("Data/raw/hh14_bus_dta.zip", "bus_us.dta", ["height", "tinggi", "cm"])

# Extract water and sanitation from KR
extract_and_explore("Data/raw/hh14_b2_dta.zip", "b2_kr.dta", ["water", "air", "toilet", "jamban", "sanitation", "wc", "feces"])

df_res = pd.DataFrame(results)
print("=== FEATURE INVENTORY ===")
print(df_res.to_string())

out_path = "Data/processed/04b_enhanced_feature_inventory.csv"
df_res.to_csv(out_path, index=False)
print(f"Inventory saved to {out_path}")

