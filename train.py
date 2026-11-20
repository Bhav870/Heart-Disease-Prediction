"""
Train and compare three models on heart.csv, then save the best one.

Dataset: "Heart Failure Prediction" (918 patients, 11 features, HeartDisease target).

Usage:  python train.py              (reads heart.csv in this folder)
        python train.py my_data.csv
"""

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = sys.argv[1] if len(sys.argv) > 1 else "heart.csv"
TARGET = "HeartDisease"
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
BINARY = ["FastingBS"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

# --- Load and clean -------------------------------------------------------
df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
df = df.drop_duplicates()  # repeated rows would leak between train and test
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    sys.exit(f"Missing columns in {DATA}: {missing}")

X = df[FEATURES].copy()
for c in CATEGORICAL:
    X[c] = X[c].astype(str).str.strip()
# In this dataset 0 means "not measured" for these two columns (172 and 1 rows).
# Turn them into NaN so the pipeline imputes them (fit on training data only).
X[["Cholesterol", "RestingBP"]] = X[["Cholesterol", "RestingBP"]].replace(0, np.nan)
y = df[TARGET].astype(int)  # 1 = heart disease

# --- Models ---------------------------------------------------------------
def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])

models = {
    "Logistic regression": LogisticRegression(max_iter=2000),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42),
    "Gradient boosting": GradientBoostingClassifier(random_state=42),
}

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", prep()), ("clf", clf)])
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr)
    pred, prob = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    rows.append({"Model": name, "CV AUC": cv_auc,
                 "Accuracy": accuracy_score(y_te, pred),
                 "Precision": precision_score(y_te, pred),
                 "Recall": recall_score(y_te, pred),
                 "F1": f1_score(y_te, pred),
                 "Test AUC": roc_auc_score(y_te, prob)})
    fitted[name] = pipe

results = pd.DataFrame(rows).set_index("Model")
best = results["CV AUC"].idxmax()  # choose on cross-validation, not on the test set
print(results.round(3), f"\nBest model: {best}")

# Importance is computed on raw columns, so one-hot columns are grouped per feature.
importance = pd.Series(
    permutation_importance(fitted[best], X_te, y_te, scoring="roc_auc",
                           n_repeats=20, random_state=42).importances_mean,
    index=FEATURES).sort_values(ascending=False)

joblib.dump({
    "model": fitted[best], "best": best, "features": FEATURES,
    "results": results, "importance": importance,
    "cm": confusion_matrix(y_te, fitted[best].predict(X_te)),
    "medians": X.groupby(y)[NUMERIC].median(),
    "data": X.assign(target=y),
}, "model.joblib")
print("Saved model.joblib")