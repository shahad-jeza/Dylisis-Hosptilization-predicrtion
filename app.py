from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np


app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.pkl')

# Load the scaler (assuming you have saved it as well)
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request.
        data = request.form.to_dict()

        # Extract features and convert them to the correct format
        features = [float(data[field]) for field in ['age', 'gender', 'diabetes', 'hypertension', 'bmi', 'albumin', 'hemoglobin']]
        
        # Convert to numpy array and scale
        input_data = np.array([features])
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)

        # Return the prediction as a JSON response
        return jsonify(prediction=int(prediction[0]))

    except Exception as e:
        return jsonify(error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
