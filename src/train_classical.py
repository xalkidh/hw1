import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_classical(X_train, y_train, X_val, y_val):
    print("Εκπαίδευση Random Forest")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Αξιολόγηση στο validation set
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    print(f"Validation ROC-AUC: {val_auc:.4f}")

    # Feature importances
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    print(f"Top 5 feature indices (importances): {top_idx}")
    print(f"Top 5 importance values: {importances[top_idx]}")

    # Αποθήκευση μοντέλου
    os.makedirs("models", exist_ok=True)
    with open("models/classical_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Αποθηκεύτηκε στο models/classical_model.pkl")

    return model