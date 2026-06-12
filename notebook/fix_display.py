with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

content = "try:\n    from IPython.display import display\nexcept ImportError:\n    display = print\n\n" + content

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
