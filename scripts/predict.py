import joblib
import cv2
from feature_extraction import extract_features

def predict_image(image_path):
    model = joblib.load("../models/svm_model.joblib")
    preproc = joblib.load("../models/preproc.joblib")
    feats = extract_features(image_path).reshape(1, -1)
    feats = preproc['scaler'].transform(feats)
    feats = preproc['pca'].transform(feats)
    pred = model.predict(feats)
    label = preproc['encoder'].inverse_transform(pred)[0]
    print(f"Predicted class: {label}")

if __name__ == "__main__":
    img_path = "../dataset/normal/1.jpg"  # change this to any test image
    predict_image(img_path)
