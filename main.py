import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(
        page_title="megaloblastic_anemia_app",
)

# Charger les modèles
modele_folder = Path(__file__).parent / "model"

model = joblib.load(modele_folder / "best_logistic_model_pipeline.pkl")

# Dictionnaire pour traduire les variables en anglais
variable_labels = {
    'glossite': 'Glossitis',
    'vs': 'ESR',
    'vgm': 'MCV',
    'paresthesies_des_membres': 'Limb Paresthesia',
    'tcmh': 'MCHC',
    'ldh': 'LDH',
    'aregen': 'Regenerative'
}

# Liste des colonnes dans l'ordre attendu par le modèle
expected_features = ['glossite', 'vs', 'vgm', 'paresthesies_des_membres', 'tcmh', 'ldh', 'aregen']

# Liste des variables continues
continuous_variables = ['vs', 'vgm', 'tcmh', 'ldh']

# Liste des variables binaires
binary_variables = ['glossite', 'paresthesies_des_membres', 'aregen']

# Interface utilisateur
st.title("Application of the Research Work: 'Machine learning models in predictive factors for megaloblastic character of macrocytic anemia'")

# Sous-titre avec le nom du professeur
st.subheader("Directed by Professor Melek Kechida")


# Saisie utilisateur pour les variables binaires (glossite, paresthesies_des_membres, aregen)
binary_inputs = {}
for var in binary_variables:
    binary_inputs[var] = st.selectbox(f"{variable_labels[var]} (Yes/No)", ['No', 'Yes'])

# Saisie utilisateur pour les variables continues (vs, vgm, tcmh, ldh)
continuous_inputs = {}
for var in continuous_variables:
    continuous_inputs[var] = st.number_input(f"Enter {variable_labels[var]} value:", min_value=0.0, step=0.1)

# Prétraitement des entrées utilisateur
input_data = {}
for var in expected_features:
    if var in binary_inputs:
        input_data[var] = 1 if binary_inputs[var] == 'Yes' else 0
    elif var in continuous_inputs:
        input_data[var] = continuous_inputs[var]

# Transformer en DataFrame
input_df = pd.DataFrame([input_data])

# Prédiction
if st.button("Predict"):
    try:
        probability = model.predict_proba(input_df)[:, 1][0]
        prediction = "Non-Megaloblastic Anemia" if probability > 0.73 else "Megaloblastic Anemia"

        st.write(f"Prediction: {prediction}")
        st.write(f"Probability of Non-Megaloblastic Anemia: {probability:.2f}")
    except ValueError as e:
        st.error(f"An error occurred: {e}")


# Séparation
st.markdown('---')

# Section "Authors"
st.subheader('Authors')
st.write("Application created by Resident Mohamed Kenani under the guidance of Professor Melek Kechida")

# Séparation avant la section Contact
st.markdown('---')
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Section "Contact"
st.subheader('Contact')
container6 = st.container()

contact_form = """
<form action="https://formsubmit.co/kenanimohamed19@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" required placeholder="Your name" required>
     <input type="email" name="email" required placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here" required></textarea>
     <button type="submit">Send</button>
</form>"""
left_column, right_column = container6.columns(2)
left_column.markdown(contact_form, unsafe_allow_html=True)
right_column.empty()