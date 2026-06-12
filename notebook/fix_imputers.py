import re

with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

cb_pipe_old = """cb_pipeline = Pipeline([
    ('model', CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=0,
        thread_count=1
    ))
])"""
cb_pipe_new = """cb_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=0,
        thread_count=1
    ))
])"""
content = content.replace(cb_pipe_old, cb_pipe_new)

lgbm_pipe_old = """lgbm_pipeline = Pipeline([
    ('model', LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1
    ))
])"""
lgbm_pipe_new = """lgbm_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1
    ))
])"""
content = content.replace(lgbm_pipe_old, lgbm_pipe_new)

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
print("Added imputers to CB and LGBM")
