import zipfile
import pyreadstat
import os

def explore_kr():
    zip_path = "Data/raw/hh14_b1_dta.zip"
    with zipfile.ZipFile(zip_path, 'r') as z:
        for fname in z.namelist():
            if "kr" in fname.lower() or "ks" in fname.lower():
                tmp_path = f"/tmp/{os.path.basename(fname)}"
                with open(tmp_path, "wb") as f:
                    f.write(z.read(fname))
                
                try:
                    _, meta = pyreadstat.read_dta(tmp_path, metadataonly=True)
                    print(f"=== {fname} ===")
                    for c, l in list(meta.column_names_to_labels.items())[:30]:  # print first 30 to see
                        print(f"  {c}: {l}")
                    print("...")
                except Exception as e:
                    print(e)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

explore_kr()
