import re

with open('notebook/stunting_paper_final.py', 'r') as f:
    content = f.read()

# 1. Update scoring dictionary
content = re.sub(
    r"scoring = \{\n    \"accuracy\": \"accuracy\",\n    \"precision\": make_scorer\(precision_score, pos_label=1, zero_division=0\),\n    \"recall\": make_scorer\(recall_score, pos_label=1\),\n    \"f1\": make_scorer\(f1_score, pos_label=1\),\n    \"roc_auc\": \"roc_auc\"\n\}",
    "scoring = {\n    \"accuracy\": \"accuracy\",\n    \"precision\": make_scorer(precision_score, pos_label=1, zero_division=0),\n    \"recall\": make_scorer(recall_score, pos_label=1),\n    \"f1\": make_scorer(f1_score, pos_label=1),\n    \"roc_auc\": \"roc_auc\"\n}\n\n# METRIK UTAMA SEKARANG: ROC-AUC\nPRIMARY_SCORING = 'roc_auc'",
    content
)

# 2. LR Pipeline (Remove SMOTE, change scoring)
content = re.sub(
    r"lr_pipeline = Pipeline\(\[\n    \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n    \('scaler', StandardScaler\(\)\),\n    \('model', LogisticRegression\(random_state=RANDOM_STATE, max_iter=2000\)\)\n\]\)",
    "lr_pipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000))\n])",
    content
)
content = content.replace("scoring='f1', random_state=RANDOM_STATE, n_jobs=1", "scoring=PRIMARY_SCORING, random_state=RANDOM_STATE, n_jobs=1")
content = content.replace("Best CV F1:", "Best CV ROC-AUC:")

# 3. RF Pipeline (Remove SMOTE)
content = re.sub(
    r"rf_pipeline = Pipeline\(\[\n    \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n    \('model', RandomForestClassifier\(random_state=RANDOM_STATE\)\)\n\]\)",
    "rf_pipeline = Pipeline([\n    ('model', RandomForestClassifier(random_state=RANDOM_STATE))\n])",
    content
)

# 4. XGB Pipeline (Remove SMOTE, update params)
content = re.sub(
    r"xgb_pipeline = Pipeline\(\[\n        \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n        \('model', XGBClassifier\(\n            random_state=RANDOM_STATE, eval_metric='logloss',\n            use_label_encoder=False\n        \)\)\n    \]\)",
    "xgb_pipeline = Pipeline([\n        ('model', XGBClassifier(\n            random_state=RANDOM_STATE, eval_metric='logloss',\n            use_label_encoder=False\n        ))\n    ])",
    content
)
content = re.sub(
    r"xgb_params = \{\n        'model__n_estimators': \[100, 200, 300\],\n        'model__max_depth': \[3, 5, 7\],\n        'model__learning_rate': \[0.01, 0.05, 0.1, 0.2\],\n        'model__subsample': \[0.7, 0.8, 0.9, 1.0\],\n        'model__scale_pos_weight': \[1, 2, 3\]\n    \}",
    "xgb_params = {\n        'model__n_estimators': [300, 500, 700],\n        'model__max_depth': [3, 4, 5],\n        'model__learning_rate': [0.01, 0.03, 0.05],\n        'model__subsample': [0.8, 0.9, 1.0],\n        'model__colsample_bytree': [0.8, 0.9, 1.0]\n    }",
    content
)

# 5. CB Pipeline (Remove SMOTE)
content = re.sub(
    r"cb_pipeline = Pipeline\(\[\n        \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n        \('model', CatBoostClassifier\(\n            random_state=RANDOM_STATE, verbose=0\n        \)\)\n    \]\)",
    "cb_pipeline = Pipeline([\n        ('model', CatBoostClassifier(\n            random_state=RANDOM_STATE, verbose=0\n        ))\n    ])",
    content
)

# 6. LGBM Pipeline (Remove SMOTE)
content = re.sub(
    r"lgb_pipeline = Pipeline\(\[\n        \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n        \('model', LGBMClassifier\(\n            random_state=RANDOM_STATE, verbose=-1\n        \)\)\n    \]\)",
    "lgb_pipeline = Pipeline([\n        ('model', LGBMClassifier(\n            random_state=RANDOM_STATE, verbose=-1\n        ))\n    ])",
    content
)

# 7. CV Section (Fix prints)
content = content.replace("Cross-Validation Ketat (SMOTE di dalam Pipeline)...", "Cross-Validation Ketat (Tanpa SMOTE)...")
content = content.replace("F1={res['test_f1'].mean():.3f}, AUC={res['test_roc_auc'].mean():.3f}", "AUC={res['test_roc_auc'].mean():.3f}, F1={res['test_f1'].mean():.3f}")

# 8. Ensemble (Remove SMOTE fit)
content = re.sub(
    r"# We need to fit models on SMOTE-resampled training data for the ensemble\nsmote = SMOTE\(random_state=RANDOM_STATE\)\nX_train_resampled, y_train_resampled = smote.fit_resample\(X_train_selected, y_train\)\n\nprint\(f\"Data training setelah SMOTE: \{X_train_resampled.shape\[0\]\} sampel\"\)\nprint\(f\"  Not Stunted: \{\(y_train_resampled == 0\).sum\(\)\}\"\)\nprint\(f\"  Stunted:     \{\(y_train_resampled == 1\).sum\(\)\}\"\)",
    "# Ensemble uses RAW selected features now\nX_train_resampled, y_train_resampled = X_train_selected, y_train",
    content
)
content = re.sub(
    r"ensemble_pipeline = Pipeline\(\[\n        \('smote', SMOTE\(random_state=RANDOM_STATE\)\),\n        \('model', ensemble\)\n    \]\)",
    "ensemble_pipeline = Pipeline([\n        ('model', ensemble)\n    ])",
    content
)
content = content.replace("Ensemble Voting CV: F1=", "Ensemble Voting CV: AUC=")

# 9. Threshold tuning (cross_val_predict)
content = content.replace("best_name = max(results, key=lambda x: (results[x][\"f1\"], results[x][\"recall\"], results[x][\"auc\"]))", "best_name = max(results, key=lambda x: (results[x][\"auc\"], results[x][\"f1\"]))")

old_threshold_code = """# Threshold Tuning
thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
tuning_rows = []
y_best_prob = best_result["y_prob"]

for threshold in thresholds:
    pred_t = (y_best_prob >= threshold).astype(int)
    tuning_rows.append({
        "Threshold": float(threshold),
        "Precision": precision_score(y_test, pred_t, zero_division=0),
        "Recall": recall_score(y_test, pred_t, zero_division=0),
        "F1-Score": f1_score(y_test, pred_t, zero_division=0)
    })"""

new_threshold_code = """# Threshold Tuning menggunakan Out-of-Fold (OOF) CV pada Training Set
from sklearn.model_selection import cross_val_predict
print("Mencari threshold optimal menggunakan Cross-Validation...")

# Get cross-validated probabilities on training set
y_oof_prob = cross_val_predict(best_pipeline, X_train_selected, y_train, cv=cv_strategy, method='predict_proba')[:, 1]

thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
tuning_rows = []

for threshold in thresholds:
    pred_t = (y_oof_prob >= threshold).astype(int)
    tuning_rows.append({
        "Threshold": float(threshold),
        "Precision": precision_score(y_train, pred_t, zero_division=0),
        "Recall": recall_score(y_train, pred_t, zero_division=0),
        "F1-Score": f1_score(y_train, pred_t, zero_division=0)
    })"""

content = content.replace(old_threshold_code, new_threshold_code)

# 10. Update test results reporting logic for the tuned threshold
old_tune_results = """best_tune = tuning_df.loc[tuning_df["F1-Score"].idxmax()]
optimal_threshold = float(best_tune["Threshold"])

print(f"\\nOptimal threshold: {optimal_threshold:.2f}")
print(f"  Precision: {best_tune['Precision']:.4f}")
print(f"  Recall:    {best_tune['Recall']:.4f}")
print(f"  F1-Score:  {best_tune['F1-Score']:.4f}")"""

new_tune_results = """best_tune = tuning_df.loc[tuning_df["F1-Score"].idxmax()]
optimal_threshold = float(best_tune["Threshold"])

print(f"\\nOptimal threshold (ditemukan dari CV Train): {optimal_threshold:.2f}")
print(f"  OOF Precision: {best_tune['Precision']:.4f}")
print(f"  OOF Recall:    {best_tune['Recall']:.4f}")
print(f"  OOF F1-Score:  {best_tune['F1-Score']:.4f}")

# Re-evaluate Test Set with Optimal Threshold
y_test_prob = best_result["y_prob"]
final_test_pred = (y_test_prob >= optimal_threshold).astype(int)

best_result['precision'] = precision_score(y_test, final_test_pred, zero_division=0)
best_result['recall'] = recall_score(y_test, final_test_pred, zero_division=0)
best_result['f1'] = f1_score(y_test, final_test_pred, zero_division=0)
best_result['accuracy'] = accuracy_score(y_test, final_test_pred)

print(f"\\nPerforma Test Set pada Threshold {optimal_threshold:.2f}:")
print(f"  Test Precision: {best_result['precision']:.4f}")
print(f"  Test Recall:    {best_result['recall']:.4f}")
print(f"  Test F1-Score:  {best_result['f1']:.4f}")
"""

content = content.replace(old_tune_results, new_tune_results)


with open('notebook/stunting_paper_final.py', 'w') as f:
    f.write(content)
