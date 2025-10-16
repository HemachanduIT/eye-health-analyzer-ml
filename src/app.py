# src/app.py
# import os
# from flask import Flask, request, render_template_string, redirect, url_for
# import joblib
# import cv2
# import argparse
# from features import extract_features_from_image

# # ---------- Command-line argument for model selection ----------
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--model_type",
#     type=str,
#     default="random_forest",  # You can set "svc_model" or "random_forest"
#     help="Choose which model to load: 'svc' or 'random_forest'"
# )
# args, unknown = parser.parse_known_args()

# # ---------- Paths ----------
# MODEL_PATH = os.path.join("..", "models", f"{args.model_type}.joblib")
# SCALER_PATH = os.path.join("..", "models", "scaler.joblib")
# LE_PATH = os.path.join("..", "models", "label_encoder.joblib")
# src/app.py
import os
import argparse
from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import cv2
from features import extract_features_from_image

# Argument parsing for model type
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="svc", choices=["svc", "random_forest"], help="Choose model type")
args = parser.parse_args()

if args.model_type == "random_forest":
    MODEL_PATH = os.path.join("..", "models", "random_forest.joblib")
else:
    MODEL_PATH = os.path.join("..", "models", "svc_model.joblib")  # ✅ corrected name

SCALER_PATH = os.path.join("..", "models", "scaler.joblib")
LE_PATH = os.path.join("..", "models", "label_encoder.joblib")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- Load model artifacts ----------
print(f"🔹 Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)
print("✅ Model and supporting files loaded successfully!")

# ---------- HTML template ----------
HTML = """
<!doctype html>
<title>Eye Health Analyzer</title>
<h1>Upload an eye image for analysis</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image required>
  <input type=submit value=Analyze>
</form>
{% if result %}
  <h2>Result: {{ result }}</h2>
  {% if confidence %}
    <p>Confidence: {{ confidence|round(4) }}</p>
  {% endif %}
  <img src="{{ url_for('uploaded_file', filename=filename) }}" style="max-width:400px;">
{% endif %}
"""

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    confidence = None
    filename = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        # Read image and extract features
        img = cv2.imread(save_path)
        feat = extract_features_from_image(img)

        # Try scaling (if model expects scaled input)
        try:
            feat_s = scaler.transform([feat])
            pred = model.predict(feat_s)[0]
            proba = model.predict_proba(feat_s).max() if hasattr(model, "predict_proba") else None
        except Exception:
            # For RandomForest, scaling may not be needed
            pred = model.predict([feat])[0]
            proba = model.predict_proba([feat]).max() if hasattr(model, "predict_proba") else None

        # Decode label
        label = le.inverse_transform([pred])[0]
        result = label
        confidence = float(proba) if proba is not None else None
        filename = file.filename

    return render_template_string(HTML, result=result, confidence=confidence, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
