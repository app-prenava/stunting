with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()
content = content.replace('df = pd.read_csv(DATA_PATH)', 'df = pd.read_csv(DATA_PATH)\ndf.drop(columns=["maternal_marriage_age"], inplace=True, errors="ignore")')
with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
