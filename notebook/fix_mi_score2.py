import json

with open("notebook/stunting_paper_final.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "mi_scores = mutual_info_classif(X_train_fe.fillna(X_train_fe.median())" in source:
            new_source = source.replace(
                "mi_scores = mutual_info_classif(X_train_fe.fillna(X_train_fe.median()), y_train, random_state=RANDOM_STATE)",
                "mi_scores = mutual_info_classif(X_train_fe.fillna(0), y_train, random_state=RANDOM_STATE)"
            )
            lines = new_source.splitlines(keepends=True)
            cell["source"] = lines

with open("notebook/stunting_paper_final.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

