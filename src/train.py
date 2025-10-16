# src/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from features import extract_features_from_image
from utils import load_images_and_labels, encode_labels, save_model
import joblib
import argparse
import matplotlib.pyplot as plt

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

    print("🔹 Loading images...")
    images, labels = load_images_and_labels(dataset_dir)
    if len(images) == 0:
        raise RuntimeError("❌ No images found. Check dataset_dir")

    print("🔹 Extracting features (this may take a while)...")
    X = build_feature_matrix(images)
    y, le = encode_labels(labels)

    print("✅ Feature matrix shape:", X.shape)
    print("✅ Classes:", list(le.classes_))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scaling for SVC
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("\n🚀 Training models...")

    # 1️⃣ Train SVC
    svc = SVC(kernel='linear', probability=True, random_state=42)
    svc.fit(X_train_s, y_train)
    y_pred_svc = svc.predict(X_test_s)
    acc_svc = accuracy_score(y_test, y_pred_svc)
    print("\n📘 SVC Results:")
    print("Accuracy:", round(acc_svc * 100, 2), "%")
    print(classification_report(y_test, y_pred_svc, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svc))
    svc_path = os.path.join(out_dir, "svc_model.joblib")
    joblib.dump(svc, svc_path)
    print("✅ Saved:", svc_path)

    # 2️⃣ Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)  # RF works well without scaling
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\n🌲 Random Forest Results:")
    print("Accuracy:", round(acc_rf * 100, 2), "%")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    rf_path = os.path.join(out_dir, "random_forest.joblib")
    joblib.dump(rf, rf_path)
    print("✅ Saved:", rf_path)

    # 3️⃣ Save scaler & label encoder (shared)
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    le_path = os.path.join(out_dir, "label_encoder.joblib")
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    print("✅ Saved scaler & label encoder")

    # 4️⃣ Compare results
    print("\n🏁 Model Comparison:")
    print(f"SVC Accuracy          : {acc_svc:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    if acc_rf > acc_svc:
        print("\n🎯 Random Forest performed better overall!")
    elif acc_rf < acc_svc:
        print("\n🎯 SVC performed better overall!")
    else:
        print("\n🤝 Both models performed equally well!")

    # 5️⃣ Optional: Plot Accuracy Comparison
    plt.bar(['SVC', 'Random Forest'], [acc_svc, acc_rf], color=['blue', 'green'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'))
    print("📊 Saved comparison chart as model_comparison.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset", help="Path to dataset folder (subfolders per class)")
    parser.add_argument("--out_dir", type=str, default="../models", help="Where to save models")
    args = parser.parse_args()
    main(args)
