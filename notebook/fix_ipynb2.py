import json

with open("notebook/stunting_paper_final.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "dominant_pct = df[col].value_counts(normalize=True).iloc[0] * 100" in source:
            new_source = source.replace(
                "dominant_pct = df[col].value_counts(normalize=True).iloc[0] * 100\n    if dominant_pct > 95:\n        print(f\"  ⚠ {col}: {dominant_pct:.1f}% bernilai sama → NEAR-CONSTANT\")",
                "counts = df[col].value_counts(normalize=True)\n    if not counts.empty:\n        dominant_pct = counts.iloc[0] * 100\n        if dominant_pct > 95:\n            print(f\"  ⚠ {col}: {dominant_pct:.1f}% bernilai sama → NEAR-CONSTANT\")"
            )
            lines = new_source.splitlines(keepends=True)
            cell["source"] = lines

with open("notebook/stunting_paper_final.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

