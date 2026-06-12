with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()
    
# Find the SHAP part
shap_old = """    # Gunakan LinearExplainer untuk Logistic Regression
    best_lr = lr_search.best_estimator_.named_steps['model']
    explainer = shap.LinearExplainer(best_lr, X_train_fe)
    shap_values = explainer.shap_values(X_test_fe)"""

shap_new = """    # Gunakan LinearExplainer untuk Logistic Regression
    best_lr = lr_search.best_estimator_.named_steps['model']
    
    # Transform data before SHAP (Impute + Scale)
    preprocessor = Pipeline(lr_search.best_estimator_.steps[:-1])
    X_train_transformed = pd.DataFrame(preprocessor.transform(X_train_fe), columns=X_train_fe.columns)
    X_test_transformed = pd.DataFrame(preprocessor.transform(X_test_fe), columns=X_test_fe.columns)
    
    explainer = shap.LinearExplainer(best_lr, X_train_transformed)
    shap_values = explainer.shap_values(X_test_transformed)"""

content = content.replace(shap_old, shap_new)

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
