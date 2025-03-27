import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

ruta_scaler = os.path.join(os.path.dirname(__file__), "scaler.joblib")
ruta_model = os.path.join(os.path.dirname(__file__), "modelo.joblib")

scaler = joblib.load(ruta_scaler)
model = joblib.load(ruta_model)

# Configuración de la aplicación
st.title("Predicción Diabetes")
st.subheader("Santiago Amado y Efraín Álvarez")

# Introducción
st.write("Esta aplicación permite predecir la probabilidad de tener diabetes basada en ciertos parámetros médicos.")

# Imagen
st.image("https://news.propatiens.com/wp-content/uploads/2020/02/sintomas-diabetes.jpg")

# Entrada de datos
pregnancies = st.slider("Embarazos", 0, 13, 1)
glucose = st.slider("Glucosa", 38, 199, 100)
blood_pressure = st.slider("Presión Sanguínea", 35, 107, 80)
skin_thickness = st.slider("Grosor de Piel", 0, 80, 40)
insulin = st.slider("Insulina", 0, 318, 79)
bmi = st.slider("Índice de Masa Corporal (BMI)", 13.4, 50.0, 25.0)
diabetes_pedigree = st.slider("Función de Pedigrí de Diabetes", 0.1, 1.2, 0.5)
age = st.slider("Edad", 21, 66, 30)

# Crear DataFrame
data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
                     columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

# Normalizar los datos
data_scaled = scaler.transform(data)

# Realizar predicción
prediction = model.predict(data_scaled)

# Mostrar resultado
st.markdown("---")
if prediction[0] == 0:
    st.text("No tiene diabetes 😊")
else:
    st.text("Tiene diabetessss 😔")


st.markdown("---")
st.text("© UNAB 2025")
