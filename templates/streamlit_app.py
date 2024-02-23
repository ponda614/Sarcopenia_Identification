import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import json

# Load model and other dependencies
scaler = joblib.load("path/to/scaler.joblib")
model = tf.keras.models.load_model("path/to/model.h5")

# Load top features
with open("path/to/top_features.json", "r") as file:
    top_features = json.load(file)

# Streamlit UI
st.title('Sarcopenia Detection Tool')

# Create input fields for each of the top features
input_data = {}
for feature in top_features:
    input_data[feature] = st.number_input(f"Enter {feature}", format="%.2f")

# Predict button
if st.button('Predict'):
    # Create DataFrame from input data
    df = pd.DataFrame([input_data])

    # Scale the input data
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)
    prediction_class = int(prediction[0][0] >= 0.5)

   # Display the result
    if prediction_class == 1:
        st.success('Positive Detection Of Sarcopenia')
    else:
        st.error('Negative Detection of Sarcopenia')
