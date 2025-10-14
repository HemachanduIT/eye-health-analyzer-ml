# Eye Health Analyzer (ML)

## Setup
1. Create a Python virtual environment and activate it:
   - `python -m venv venv`
   - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

2. Install requirements:
   - `pip install -r requirements.txt`

3. Prepare dataset:
   - Put images in `dataset/<class_name>/*.jpg` (for example `dataset/normal/xxx.jpg`, `dataset/abnormal/yyy.jpg`)

## Train
Run training to extract features and train models:
<-------------------------------------------------------------->
python src/train.py --dataset_dir dataset --out_dir models

This will:
- Extract features (HOG, LBP, GLCM, color histograms),
- Train SVM and RandomForest,
- Pick the best model by test accuracy,
- Save model, scaler, and label encoder into `models/`.

## Inference (single image)
<-------------------------------------------------------------------->
python src/infer.py --image path/to/image.jpg --model models/random_forest.joblib --scaler models/scaler.joblib --le models/label_encoder.joblib

## Web UI
Run:
python src/app.py

<---------------------------------------------------------------------->
Open `http://127.0.0.1:5000/` and upload an image
## Notes
- Tweak feature parameters in `src/features.py` if you want different behavior.
- If training is slow, cache feature arrays (save `X.npy`, `y.npy`) after extraction and skip repeated extraction.
- Add unit tests in `tests/` as needed.
## Data set download link
Dataset link:
👉 https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

Description:
This dataset contains labeled eye disease images categorized into multiple classes, typically like:

Cataract

Diabetic Retinopathy

Glaucoma

Normal Eye

Others (sometimes)

Each folder contains images for one specific condition.
