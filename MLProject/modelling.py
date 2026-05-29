"""
modelling.py
============
Kriteria Basic - Melatih model machine learning dengan MLflow autolog
Dataset : heart_disease_uci_preprocessing.csv  (sudah melalui preprocessing)
Model   : Random Forest Classifier (Scikit-Learn)
Logging : MLflow autolog (tanpa hyperparameter tuning)
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Konfigurasi MLflow
EXPERIMENT_NAME = "Heart_Disease_Classification_Basic"
# mlflow.set_tracking_uri("mlruns")          # simpan lokal di folder mlruns
mlflow.set_experiment(EXPERIMENT_NAME)

# Load dataset preprocessed
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_uci_preprocessing.csv")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded  : {df.shape[0]} rows x {df.shape[1]} cols")

# Binarisasi target  (0 = tidak sakit, 1 = sakit)
df["target"] = (df["num"] > 0).astype(int)
df = df.drop(columns=["num"])

X = df.drop(columns=["target"])
y = df["target"]

print(f"Features        : {list(X.columns)}")
print(f"Target dist     :\n{y.value_counts()}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train size       : {X_train.shape[0]}")
print(f"Test  size       : {X_test.shape[0]}")

# MLflow autolog + Training
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False,
)

with mlflow.start_run(run_name="RandomForest_Basic_Autolog") as run:
    print(f"\n[MLflow] Run ID : {run.info.run_id}")
    print(f"[MLflow] Experiment : {EXPERIMENT_NAME}")

    # ── Default hyperparameters (tanpa tuning) ──
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    # autolog akan merekam semua metric & param secara otomatis
    model.fit(X_train, y_train)

    # Evaluasi manual (untuk verifikasi)
    from sklearn.metrics import accuracy_score, classification_report
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy   : {acc:.4f}")
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))

# Run selesai. Buka MLflow UI dengan:
#    mlflow ui --backend-store-uri mlruns
#    lalu buka http://127.0.0.1:5000