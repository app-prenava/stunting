import json

# Fix stunting_paper_final.py first
with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

# Make it safe
safe_code = """    counts = df[col].value_counts(normalize=True)
    if not counts.empty:
        dominant_pct = counts.iloc[0] * 100
        if dominant_pct > 95:
            print(f"  {col}: {dominant_pct:.1f}% bernilai sama -> NEAR-CONSTANT")"""

old_code = """    dominant_pct = df[col].value_counts(normalize=True).iloc[0] * 100
    if dominant_pct > 95:
        print(f"  {col}: {dominant_pct:.1f}% bernilai sama -> NEAR-CONSTANT")"""

content = content.replace(old_code, safe_code)

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)

# Use Jupytext to sync back to ipynb
import subprocess
subprocess.run(["jupytext", "--to", "ipynb", "notebook/stunting_paper_final.py", "--update", "--output", "notebook/stunting_paper_final.ipynb"])

