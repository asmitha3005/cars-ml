from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model & encoders
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()

        data = []
        for col in input_data:
            le = encoders[col]
            value = le.transform([input_data[col]])[0]
            data.append(value)

        final_input = np.array([data])
        prediction = model.predict(final_input)

        # Decode output
        output = encoders['class'].inverse_transform(prediction)[0]

        return render_template('index.html', prediction_text=f"Prediction: {output}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)