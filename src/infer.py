# src/infer.py
import argparse
import cv2
import numpy as np
import joblib
from features import extract_features_from_image
from utils import load_model
import os

def predict_image(model_path, scaler_path, le_path, image_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image: " + image_path)

    feat = extract_features_from_image(img, resize_to=(256,256))
    # if model expects scaled features
    try:
        feat_scaled = scaler.transform([feat])
    except Exception:
        # scaler may not be needed (model is RF) -> try without
        feat_scaled = None

    # try both
    if feat_scaled is not None:
        pred = model.predict(feat_scaled)[0]
        proba = model.predict_proba(feat_scaled).max() if hasattr(model, "predict_proba") else None
    else:
        pred = model.predict([feat])[0]
        proba = model.predict_proba([feat]).max() if hasattr(model, "predict_proba") else None

    label = le.inverse_transform([pred])[0]
    return label, float(proba) if proba is not None else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../models/random_forest.joblib")
    parser.add_argument("--scaler", type=str, default="../models/scaler.joblib")
    parser.add_argument("--le", type=str, default="../models/label_encoder.joblib")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    label, proba = predict_image(args.model, args.scaler, args.le, args.image)
    print("Prediction:", label)
    if proba is not None:
        print("Confidence:", proba)
