import zipfile
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

raw_dir = Path('Data/raw')
keywords = ['weight', 'length', 'breastfeed', 'asi ', 'birth', 'preg', 'marri', 'income', 'expend', 'floor', 'wall', 'water', 'toilet', 'diarrhea', 'ispa', 'anemia', 'hb ', 'hemoglobin', 'bmi', 'nutri', 'miscarri']

for zfile in raw_dir.glob('*.zip'):
    with zipfile.ZipFile(zfile, 'r') as z:
        for fname in z.namelist():
            if fname.endswith('.dta'):
                with z.open(fname) as f:
                    try:
                        iterator = pd.read_stata(f, iterator=True)
                        labels = iterator.variable_labels()
                        found = False
                        for var, label in labels.items():
                            label_lower = label.lower()
                            if any(k in label_lower for k in keywords):
                                if not found:
                                    print(f"\n--- {fname} ---")
                                    found = True
                                print(f"{var}: {label}")
                    except Exception as e:
                        pass
