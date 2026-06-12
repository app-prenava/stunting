with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()
content = content.replace(
    'mi_scores = mutual_info_classif(X_train_fe, y_train, random_state=RANDOM_STATE)',
    'mi_scores = mutual_info_classif(X_train_fe.fillna(X_train_fe.median()), y_train, random_state=RANDOM_STATE)'
)
with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
