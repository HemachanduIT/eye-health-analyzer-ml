import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from feature_extraction import extract_features

def build_dataset(dataset_dir, max_files=None):
    X, y = [], []
    for label in sorted(os.listdir(dataset_dir)):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg','.png'))]
        if max_files:
            files = files[:max_files]
        for file in tqdm(files, desc=f"Processing {label}"):
            try:
                features = extract_features(os.path.join(folder, file))
                X.append(features)
                y.append(label)
            except:
                pass
    return np.array(X), np.array(y)

def main():
    dataset_path = "../dataset"  # adjust if needed
    X, y = build_dataset(dataset_path)
    print("Feature matrix shape:", X.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.95)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    print("Training SVM...")
    param_grid = {'C':[1,10], 'gamma':['scale','auto'], 'kernel':['rbf']}
    svm = GridSearchCV(SVC(probability=True), param_grid, cv=3)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    print("Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("RF Accuracy:", accuracy_score(y_test, y_pred_rf))

    joblib.dump({'scaler':scaler, 'pca':pca, 'encoder':le}, "../models/preproc.joblib")
    joblib.dump(svm.best_estimator_, "../models/svm_model.joblib")
    joblib.dump(rf, "../models/rf_model.joblib")
    print("Models saved successfully!")

if __name__ == "__main__":
    main()
