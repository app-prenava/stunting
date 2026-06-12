import json

with open("notebook/stunting_paper_final.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "dominant_pct = X_train_fe[col].value_counts(normalize=True).iloc[0]" in source:
            new_source = source.replace(
                "dominant_pct = X_train_fe[col].value_counts(normalize=True).iloc[0]",
                "counts = X_train_fe[col].value_counts(normalize=True)\n    if counts.empty:\n        continue\n    dominant_pct = counts.iloc[0]"
            )
            lines = new_source.splitlines(keepends=True)
            cell["source"] = lines

with open("notebook/stunting_paper_final.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

