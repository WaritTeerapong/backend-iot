import logging

import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO)

# Model definition
class SignLanguageModel(nn.Module):
    def __init__(self, input_size=11, hidden_size=128, output_size=5):
        super(SignLanguageModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "best.pt"  # Ensure this is the correct path to your model
model = SignLanguageModel(input_size=11).to(device)

# Correctly load the model state dictionary
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    raise

# Flask setup
app = Flask(__name__)
CORS(app)

# Label mapping
label_mapping = {0: "susu", 1: "Love", 2: "Good", 3: "Bad", 4: "Gun"}

# Global variable to store the latest prediction
latest_prediction = {"predicted_word": None, "confidence": None}

@app.route('/predict', methods=['POST'])
def custom_prediction():
    """Handle custom input data in comma-separated format."""
    global latest_prediction
    try:
        data = request.json
        if 'input_data' not in data:
            raise ValueError("Missing 'input_data' in the request.")

        input_data = list(map(float, data['input_data'].split(',')))

        if len(input_data) != 11:
            raise ValueError("Input data must contain exactly 11 values.")

        prediction = run_prediction(input_data)
        latest_prediction = prediction  # Update latest prediction
        return jsonify(prediction)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

def run_prediction(input_data):
    """Common prediction logic."""
    try:
        input_tensor = torch.tensor([input_data], dtype=torch.float32).to(device).unsqueeze(1)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            confidence = round(confidence.max().item(), 2)

        predicted_word = label_mapping[int(predicted.item())]
        logging.info(f"Prediction: {predicted_word}, Confidence: {confidence}")
        return {
            "predicted_word": predicted_word,
            "confidence": confidence,
        }
    except Exception as e:
        logging.error(f"Error in prediction logic: {e}")
        return {"error": str(e)}

@app.route('/latest', methods=['GET'])
def get_latest_prediction():
    """Send the latest prediction to the frontend."""
    global latest_prediction
    if latest_prediction["predicted_word"] is None:
        return jsonify({"error": "No predictions available yet"}), 404
    return jsonify(latest_prediction)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
