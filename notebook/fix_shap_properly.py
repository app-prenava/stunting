import re
with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

shap_prep_old = """    # Need to get resampled + scaled data
    if best_scaler is not None:
        X_train_shap = best_scaler.transform(X_train_resampled)
    else:
        X_train_shap = X_train_resampled
    explainer = shap.LinearExplainer(shap_model, X_train_shap)
    X_test_shap = best_scaler.transform(X_test_selected) if best_scaler else X_test_selected"""

shap_prep_new = """    # Need to get resampled + scaled data
    from sklearn.pipeline import Pipeline
    preprocessor = Pipeline(best_pipeline.steps[:-1])
    X_train_shap = pd.DataFrame(preprocessor.transform(X_train_resampled), columns=X_train_resampled.columns)
    explainer = shap.LinearExplainer(shap_model, X_train_shap)
    X_test_shap = pd.DataFrame(preprocessor.transform(X_test_selected), columns=X_test_selected.columns)"""

content = content.replace(shap_prep_old, shap_prep_new)

# Also fix CatBoost / LightGBM TreeExplainer
shap_tree_old = """else:
    explainer = shap.TreeExplainer(shap_model)
    X_test_shap = X_test_selected"""

shap_tree_new = """else:
    from sklearn.pipeline import Pipeline
    preprocessor = Pipeline(best_pipeline.steps[:-1])
    X_test_shap = pd.DataFrame(preprocessor.transform(X_test_selected), columns=X_test_selected.columns)
    explainer = shap.TreeExplainer(shap_model)"""

content = content.replace(shap_tree_old, shap_tree_new)

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
