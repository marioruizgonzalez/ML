import streamlit as st
import joblib
import numpy as np

# Cargar el modelo entrenado
model = joblib.load('diabetes_regression_model.pkl')

# Definir las entradas (features) del usuario
pregnancies = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
blood_pressure = st.number_input('Blood Pressure')
skin_thickness = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

# Predecir la progresión de la diabetes
if st.button('Predict'):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, 0, 0]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Diabetes Progression: {prediction[0]}")
