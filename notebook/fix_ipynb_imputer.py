import json

with open("notebook/stunting_paper_final.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # Fix Logistic Regression Pipeline
        if "lr_pipeline = Pipeline([\n    ('scaler', StandardScaler())," in source:
            new_source = source.replace(
                "lr_pipeline = Pipeline([\n    ('scaler', StandardScaler()),",
                "from sklearn.impute import SimpleImputer\nlr_pipeline = Pipeline([\n    ('imputer', SimpleImputer(strategy='median')),\n    ('scaler', StandardScaler()),"
            )
            cell["source"] = new_source.splitlines(keepends=True)
            source = new_source # Update for next replacement

        # Fix Random Forest Pipeline
        if "rf_pipeline = Pipeline([\n    ('model', RandomForestClassifier" in source:
            new_source = source.replace(
                "rf_pipeline = Pipeline([\n    ('model', RandomForestClassifier",
                "from sklearn.impute import SimpleImputer\nrf_pipeline = Pipeline([\n    ('imputer', SimpleImputer(strategy='median')),\n    ('model', RandomForestClassifier"
            )
            cell["source"] = new_source.splitlines(keepends=True)

with open("notebook/stunting_paper_final.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

