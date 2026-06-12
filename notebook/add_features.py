with open("notebook/stunting_paper_final.py", "r") as f:
    content = f.read()

labels_old = """    "maternal_miscarriages": "Maternal Miscarriages",
    "maternal_parity": "Maternal Parity",
    "low_birth_weight": "Low Birth Weight (<2.5kg)",
    "maternal_anemia": "Maternal Anemia (Hb<11)",
    "maternal_bmi_underweight": "Mother Underweight (BMI<18.5)",
    "maternal_bmi_overweight": "Mother Overweight (BMI>=25)",
    "adequate_anc": "Adequate ANC (>=4 Visits)",
    "food_expenditure_quintile": "Food Expenditure Quintile",
    "child_breastfed": "Breastfed",
    "child_had_diarrhea": "Child had Diarrhea",
    "child_had_ari": "Child had ARI",
"""

labels_new = """    "maternal_miscarriages": "Maternal Miscarriages",
    "maternal_parity": "Maternal Parity",
    "low_birth_weight": "Low Birth Weight (<2.5kg)",
    "maternal_anemia": "Maternal Anemia (Hb<11)",
    "maternal_bmi_underweight": "Mother Underweight (BMI<18.5)",
    "maternal_bmi_overweight": "Mother Overweight (BMI>=25)",
    "adequate_anc": "Adequate ANC (>=4 Visits)",
    "food_expenditure_quintile": "Food Expenditure Quintile",
    "child_breastfed": "Breastfed",
    "child_had_diarrhea": "Child had Diarrhea",
    "child_had_ari": "Child had ARI",
    "age_in_months": "Child Age (Months)",
    "maternal_bmi": "Maternal BMI",
    "maternal_hemoglobin": "Maternal Hemoglobin",
    "child_birth_weight": "Birth Weight (kg)",
    "maternal_anc_visits": "ANC Visits",
    "hh_food_expenditure_weekly": "Weekly Food Expenditure",
"""

content = content.replace(labels_old, labels_new)

with open("notebook/stunting_paper_final.py", "w") as f:
    f.write(content)
