from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from flask_cors import CORS
import os

print("🔥 Server starting...")

# Flask setup
app = Flask(__name__, template_folder='templates')
CORS(app)

# Load model
MODEL_PATH = os.path.join(os.getcwd(), 'crop_model.pkl')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = pickle.load(open(MODEL_PATH, 'rb'))
print("✅ Model loaded")

# Home route (loads your frontend)
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API (MATCHES YOUR FRONTEND EXACTLY)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert input safely
        features = np.array([[ 
            float(data.get('N', 0)),
            float(data.get('P', 0)),
            float(data.get('K', 0)),
            float(data.get('temperature', 0)),
            float(data.get('humidity', 0)),
            float(data.get('ph', 0)),
            float(data.get('rainfall', 0))
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "crop": prediction   # ✅ EXACTLY what frontend needs
        })

    except Exception as e:
        return jsonify({
            "crop": "error",
            "message": str(e)
        })


# Run server
if __name__ == '__main__':
    print("🚀 Running at http://127.0.0.1:5000")
    app.run(debug=True)