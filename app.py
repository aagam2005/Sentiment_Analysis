"""
app.py
Flask server that connects TensorFlow to a web frontend for real-time
emotion detection from text.
"""

from flask import Flask, request, jsonify, render_template
from predict import predict_emotion, load_artifacts

app = Flask(__name__)

# Load model/tokenizer once at startup (not per-request)
try:
    load_artifacts()
    print("Model, tokenizer, and labels loaded successfully.")
except FileNotFoundError as e:
    print(f"WARNING: {e}")
    print("The server will start, but /predict will fail until you run train.py.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide non-empty 'text'."}), 400

    try:
        result = predict_emotion(text)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
