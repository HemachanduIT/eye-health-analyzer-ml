# src/app.py
import os
from flask import Flask, request, render_template_string, redirect, url_for
import joblib
import cv2
from features import extract_features_from_image

MODEL_PATH = os.path.join("..", "models", "random_forest.joblib")
SCALER_PATH = os.path.join("..", "models", "scaler.joblib")
LE_PATH = os.path.join("..", "models", "label_encoder.joblib")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load artifacts once
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)

HTML = """
<!doctype html>
<title>Eye Health Analyzer</title>
<h1>Upload an eye image</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=image>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h2>Result: {{ result }}</h2>
  {% if confidence %}
    <p>Confidence: {{ confidence }}</p>
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
        img = cv2.imread(save_path)
        feat = extract_features_from_image(img)
        try:
            feat_s = scaler.transform([feat])
            pred = model.predict(feat_s)[0]
            proba = model.predict_proba(feat_s).max() if hasattr(model, "predict_proba") else None
        except Exception:
            pred = model.predict([feat])[0]
            proba = model.predict_proba([feat]).max() if hasattr(model, "predict_proba") else None

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
