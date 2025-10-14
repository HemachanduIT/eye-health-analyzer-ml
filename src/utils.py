# src/utils.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

def load_image_paths(dataset_dir):
    """
    Expects dataset_dir to have subfolders per class.
    Returns two lists: image_paths, labels
    """
    image_paths = []
    labels = []
    for label in sorted(os.listdir(dataset_dir)):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
                image_paths.append(os.path.join(label_path, fname))
                labels.append(label)
    return image_paths, labels

def load_images_and_labels(dataset_dir, max_images_per_class=None, img_read_flag=cv2.IMREAD_COLOR):
    image_paths, labels = load_image_paths(dataset_dir)
    X = []
    y = []
    count_per_class = {}
    for p, lbl in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Loading images"):
        count_per_class.setdefault(lbl,0)
        if max_images_per_class and count_per_class[lbl] >= max_images_per_class:
            continue
        img = cv2.imread(p, img_read_flag)
        if img is None:
            print("Warning: failed to read:", p)
            continue
        X.append(img)
        y.append(lbl)
        count_per_class[lbl] += 1
    return X, y

def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le

def save_model(obj, path):
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
