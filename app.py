import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = [
            request.form.get("Resturant_ID"),
            request.form.get("City"),
            request.form.get("Cuisines"),
            request.form.get("Has_table_booking"),
            request.form.get("Votes"),
            request.form.get("Avg_Cost_for_2"),
        ]

        # Convert data into a NumPy array
        data = np.array(data, dtype=float).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(data)[0]

        return render_template('index.html', prediction_text=f'Predicted Rating: {prediction:.2f}')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
