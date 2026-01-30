import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "iris_model.pkl"  # make sure this file exists

    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            return pickle.load(file)
    else:
        st.error(f"Model file '{model_path}' not found!")
        return None

model = load_model()

# ---------------- UI ----------------
st.title("🌸 Iris Species Predictor")

st.write("Enter flower measurements:")

sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width  = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width  = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    if model is not None:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)

        species = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"Predicted Species: **{species[prediction[0]]}**")
    else:
        st.warning("Model not loaded.")