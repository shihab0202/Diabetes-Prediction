import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load the trained model
model_path = 'pickel_model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index_new.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict diabetes based on input features
    Expected JSON format:
    {
        "pregnancies": float,
        "glucose": float,
        "blood_pressure": float,
        "skin_thickness": float,
        "insulin": float,
        "bmi": float,
        "diabetes_pedigree": float,
        "age": float
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('blood_pressure', 0)),
            float(data.get('skin_thickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('diabetes_pedigree', 0)),
            float(data.get('age', 0))
        ]
        
        # Convert to numpy array and reshape for prediction
        features_array = np.array([features])
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Get prediction probability if available
        try:
            probability = model.predict_proba(features_array)[0]
            confidence = float(max(probability)) * 100
        except:
            confidence = None
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_text': 'ডায়াবেটিস আছে' if prediction == 1 else 'ডায়াবেটিস নেই',
            'confidence': confidence
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/info', methods=['GET'])
def info():
    """Return information about the model"""
    return jsonify({
        'model_type': 'Diabetes Prediction Model',
        'features': [
            'Pregnancies',
            'Glucose',
            'Blood Pressure',
            'Skin Thickness',
            'Insulin',
            'BMI',
            'Diabetes Pedigree Function',
            'Age'
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
