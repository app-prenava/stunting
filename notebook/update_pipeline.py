import re

with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

# Add new feature labels
new_labels = """
    "maternal_miscarriages": "Maternal Miscarriages",
    "maternal_parity": "Maternal Parity",
    "maternal_bmi": "Maternal BMI",
    "maternal_hemoglobin": "Maternal Hemoglobin",
    "child_birth_weight": "Birth Weight (kg)",
    "child_breastfed": "Breastfed",
    "maternal_anc_visits": "ANC Visits",
    "child_had_diarrhea": "Child had Diarrhea",
    "child_had_ari": "Child had ARI",
    "hh_food_expenditure_weekly": "Weekly Food Expenditure",
"""
content = content.replace('"asset_score": "Asset Score",\n}', '"asset_score": "Asset Score",\n' + new_labels + '}')

# Make sure we drop maternal_marriage_age since it's all empty
drop_line = 'df.drop(columns=["is_high_risk_mother_age"], inplace=True, errors="ignore")'
new_drop_line = drop_line + '\n    df.drop(columns=["maternal_marriage_age"], inplace=True, errors="ignore")'
content = content.replace(drop_line, new_drop_line)

# Handle imputation in pipeline
# We need to impute all missing values before model training.
# Previously, we just used StandardScaler, which doesn't impute. Let's add SimpleImputer to the pipeline.
pipeline_import = "from sklearn.linear_model import LogisticRegression"
new_pipeline_import = pipeline_import + "\nfrom sklearn.impute import SimpleImputer"
content = content.replace(pipeline_import, new_pipeline_import)

lr_pipe_old = """lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # Menggunakan solver liblinear agar mensupport L1 (Lasso) dan L2 (Ridge) penalty
    ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear', class_weight='balanced'))
])"""
lr_pipe_new = """lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    # Menggunakan solver liblinear agar mensupport L1 (Lasso) dan L2 (Ridge) penalty
    ('model', LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear', class_weight='balanced'))
])"""
content = content.replace(lr_pipe_old, lr_pipe_new)

rf_pipe_old = """rf_pipeline = Pipeline([
    ('model', RandomForestClassifier(random_state=RANDOM_STATE))
])"""
rf_pipe_new = """rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE))
])"""
content = content.replace(rf_pipe_old, rf_pipe_new)

xgb_pipe_old = """xgb_pipeline = Pipeline([
    ('model', XGBClassifier(
        random_state=RANDOM_STATE, 
        eval_metric='logloss',
        use_label_encoder=False
    ))
])"""
xgb_pipe_new = """xgb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', XGBClassifier(
        random_state=RANDOM_STATE, 
        eval_metric='logloss',
        use_label_encoder=False
    ))
])"""
content = content.replace(xgb_pipe_old, xgb_pipe_new)

# Same for VotingClassifier
voting_old = """ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_pipeline),
        ('cb', cb_pipeline),
        ('lgbm', lgbm_pipeline)
    ],
    voting='soft'
)"""
voting_new = """ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_pipeline),
        ('cb', cb_pipeline),
        ('lgbm', lgbm_pipeline)
    ],
    voting='soft'
)"""
# wait, catboost and lgbm handle NaNs natively, but if we pass it through voting it might be fine.
# we'll let CatBoost and LightGBM handle NaNs natively. Wait, if XGB has imputer, it works. Let's just add imputer to ensemble_pipeline if we need it. But CatBoost handles NaNs. So we don't add imputer globally.

with open("notebook/stunting_paper_final.py", "r") as f:
    old_content = f.read()

if old_content != content:
    with open("notebook/stunting_paper_final.py", "w") as f:
        f.write(content)
    print("Updated pipeline scripts with imputer and new labels!")
else:
    print("No changes made.")

