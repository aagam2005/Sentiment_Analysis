# Emotion Detection From Text — RNN + Flask + TensorFlow

## Project structure
```
Sentiment_Analysis/
├── app.py              # Flask server (serves UI + /predict API)
├── train.py             # Builds & trains the Bi-LSTM model
├── data_gen.py           # Downloads + preprocesses the dataset
├── predict.py            # Loads trained model, runs inference
├── requirements.txt
├── templates/index.html
├── static/style.css
├── static/app.js
└── model/               # created automatically: dataset, tokenizer, trained model
```

## Dataset
`dair-ai/emotion` (Hugging Face) — ~20k labeled English sentences across
6 classes: sadness, joy, love, anger, fear, surprise. Downloaded automatically
by `data_gen.py`, no manual download needed.

## Setup (run these in your `Sentiment_Analysis` folder, in order)

### 1. Create and activate a virtual environment
Windows:
```
python -m venv venv
venv\Scripts\activate
```
macOS/Linux:
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify TensorFlow / Numpy / Pandas are working
```
python -c "import tensorflow as tf, numpy, pandas; print('TF:', tf.__version__); print('NumPy:', numpy.__version__); print('Pandas:', pandas.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 4. Generate/prepare the dataset (downloads + tokenizes)
```
python data_gen.py
```
This creates `model/dataset.npz`, `model/tokenizer.pickle`, `model/labels.pickle`.

### 5. Train the model
```
python train.py
```
This creates `model/emotion_model.keras`. Training ~15 epochs on this dataset
typically takes a few minutes on CPU.

### 6. (Optional) Quick CLI test
```
python predict.py
```

### 7. Run the Flask app
```
python app.py
```
Then open **http://localhost:5000** in your browser. Type a sentence and
click Analyze.

## Notes
- If you later want a React frontend instead of vanilla JS: keep `app.py`'s
  `/predict` endpoint as-is (it's a plain JSON API), scaffold a React app
  separately (`npx create-react-app frontend`), and have it `fetch()` the
  same `/predict` endpoint — you'll just need to enable CORS
  (`pip install flask-cors` and `CORS(app)` in `app.py`) since React would
  run on a different port during development.
- To retrain from scratch, delete the `model/` folder and repeat steps 4–5.
