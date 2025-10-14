# src/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import extract_features_from_image
from utils import load_images_and_labels, encode_labels, save_model
import joblib
import argparse

def build_feature_matrix(images):
    feats = []
    for img in images:
        f = extract_features_from_image(img, resize_to=(256,256))
        feats.append(f)
    return np.vstack(feats)

def main(args):
    dataset_dir = args.dataset_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print("Loading images...")
    images, labels = load_images_and_labels(dataset_dir)
    if len(images) == 0:
        raise RuntimeError("No images found. Check dataset_dir")

    print("Extracting features (this may take a while)...")
    X = build_feature_matrix(images)
    y, le = encode_labels(labels)

    print("Feature matrix shape:", X.shape)
    print("Labels:", set(labels))

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # classifiers
    print("Training models...")

    svc = SVC(kernel='linear', probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    y_pred_svc = svc.predict(X_test_s)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print("SVC accuracy:", acc_svc)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)  # RF often works well without scaling
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("RandomForest accuracy:", acc_rf)

    # choose best
    if acc_rf >= acc_svc:
        best_model = rf
        best_name = "random_forest"
    else:
        best_model = svc
        best_name = "svc"

    print("Best model:", best_name)

    # evaluation report for chosen model
    y_pred = best_model.predict(X_test_s if best_name == "svc" else X_test)
    print("Classification report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # save artifacts
    model_path = os.path.join(out_dir, f"{best_name}.joblib")
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    le_path = os.path.join(out_dir, "label_encoder.joblib")

    save_model(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)

    print("Saved model to:", model_path)
    print("Saved scaler to:", scaler_path)
    print("Saved label encoder to:", le_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset", help="Path to dataset folder (subfolders per class)")
    parser.add_argument("--out_dir", type=str, default="../models", help="Where to save models")
    args = parser.parse_args()
    main(args)
