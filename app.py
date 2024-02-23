import logging
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Load your model and other dependencies
scaler = joblib.load("C:/Users/User/Desktop/Mangimind Data Science Bootcamp/Sarcopenia_Identification/scaler.joblib")
model = tf.keras.models.load_model("C:/Users/User/Desktop/Mangimind Data Science Bootcamp/Sarcopenia_Identification/best_model_with_selected_features.h5")

# Load top features
with open("C:/Users/User/Desktop/Mangimind Data Science Bootcamp/Sarcopenia_Identification/top_features.json", "r") as file:
    top_features = json.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sarcopenia-detection')
def sarcopenia_detection():
    return render_template('sarcopenia-detection.html')

@app.route('/Sarcopenia-Patient-Analysis')
def sarcopenia_patient_analysis():
    return render_template('Sarcopenia-Patient-Analysis.html')

@app.route('/educational-resources')
def educational_resources():
    return render_template('educational-resources.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/predict-sarcopenia', methods=['POST'])  # Updated to match AJAX request URL
def predict():
    data = request.form
    
    # Convert form data to DataFrame
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric)  # Convert data to numeric type

    # Select only the top 5 features
    df_selected = df[top_features]

    # Apply scaling
    df_scaled = scaler.transform(df_selected)

    # Make prediction
    prediction = model.predict(df_scaled)

    # Adjust the threshold as needed
    threshold = 0.5
    final_prediction = int(prediction[0][0] >= threshold)

    # Customize the response message based on the prediction
    if final_prediction == 1:
        message = 'Positive Detection Of Sarcopenia'
    else:
        message = 'Negative Detection of Sarcopenia'

    return message  # Returning the custom message

if __name__ == '__main__':
    app.run(debug=True)