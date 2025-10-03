# src/baseline_model.py  — Random Forest baseline (copy-paste)
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import joblib

CLEAN_CSV = "data/processed/credit_clean.csv"

if __name__ == "__main__":
    # 1) Load processed data (from prepare_data.py)
    df = pd.read_csv(CLEAN_CSV)

    # Features (must exist in credit_clean.csv)
    features = ["LIMIT_BAL", "AGE", "EDUCATION", "MARRIAGE", "UTIL_RATIO", "PAY_STAT_AVG"]
    X = df[features]
    y = df["DEFAULT"]

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) Model: Random Forest (class-balanced)
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4) Probabilities & quick 0.5 threshold view
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    print("\n=== Threshold = 0.5 ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_05), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_05))
    print(classification_report(y_test, y_pred_05, digits=4))

    # 5) Ranking metrics (threshold-independent)
    roc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)  # PR-AUC for default class
    print(f"ROC-AUC: {roc:.4f} | PR-AUC (class 1): {pr_auc:.4f}")

    # 6) Threshold sweep to choose an operating point
    def eval_at_threshold(t):
        yp = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yp).ravel()
        prec1 = tp / (tp + fp) if (tp + fp) else 0.0
        rec1  = tp / (tp + fn) if (tp + fn) else 0.0
        f1_1  = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1) else 0.0
        acc   = (tp + tn) / (tp + tn + fp + fn)
        return acc, prec1, rec1, f1_1

    thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    for t in thresholds:
        acc, p1, r1, f1 = eval_at_threshold(t)
        rows.append({"thr": round(t,3), "acc": round(acc,4), "prec1": round(p1,4),
                     "recall1": round(r1,4), "f1_1": round(f1,4)})
    grid = pd.DataFrame(rows)
    print("\n=== Threshold sweep (acc, precision/recall/F1 for class 1) ===")
    print(grid)

    # 7) Pick threshold = best F1 for class 1 (balanced default detection)
    best_row = grid.iloc[grid["f1_1"].idxmax()]
    t = float(best_row["thr"])

    y_pred_t = (y_proba >= t).astype(int)
    print(f"\n=== Final @ threshold = {t:.3f} (best F1 class 1) ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred_t), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_t))
    print(classification_report(y_test, y_pred_t, digits=4))

    # 8) Feature importances (helps explain model)
    imps = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(imps)

    # 9) Save artifact for Streamlit
    artifact = {"model": model, "features": features, "threshold": t}
    joblib.dump(artifact, "data/model_rf.joblib")
    print("\n💾 Saved model artifact -> data/model_rf.joblib")
