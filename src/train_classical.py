import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

def train_classical(X_train, y_train, X_val, y_val, feature_names):
    results = {}

    # 1. Random Forest
    print("Εκπαίδευση Random Forest")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_pred = rf.predict(X_val)
    rf_auc = roc_auc_score(y_val, rf_proba)
    rf_f1 = f1_score(y_val, rf_pred)
    print(f"RF - Validation ROC-AUC: {rf_auc:.4f} | F1: {rf_f1:.4f}")
    
    # Feature Importances (Random Forest)
    print("Feature Importances - Random Forest")
    rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"Top 5 Features (RF):\n{rf_importances.head(5)}\n")
    
    results["rf"] = {"model": rf, "auc": rf_auc, "f1": rf_f1, "importances": rf_importances}

    # 2. XGBoost
    print("Εκπαίδευση XGBoost")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_pred = xgb.predict(X_val)
    xgb_auc = roc_auc_score(y_val, xgb_proba)
    xgb_f1 = f1_score(y_val, xgb_pred)
    print(f"XGB - Validation ROC-AUC: {xgb_auc:.4f} | F1: {xgb_f1:.4f}")
    
    # Feature Importances (XGBoost)
    print("Feature Importances - XGBoost")
    xgb_importances = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"Top 5 Features (XGB):\n{xgb_importances.head(5)}\n")
    
    results["xgb"] = {"model": xgb, "auc": xgb_auc, "f1": xgb_f1, "importances": xgb_importances}

    # 3. Οπτικοποίηση (Χρήσιμο για το README)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    rf_importances.head(10).plot(kind='barh', title='RF Top 10 Features', color='#F5277D')
    plt.subplot(1, 2, 2)
    xgb_importances.head(10).plot(kind='barh', title='XGB Top 10 Features', color='#FF6EED')
    plt.tight_layout()
    plt.show()

    # Αποθήκευση και των δύο
    os.makedirs("models", exist_ok=True)
    with open("models/classical_model_rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open("models/classical_model_xgb.pkl", "wb") as f:
        pickle.dump(xgb, f)

    # Εύρεση καλύτερου μοντέλου
    if xgb_f1 >= rf_f1:
        print("\nΚαλύτερο classical model: XGBoost")
        best = xgb
    else:
        print("\nΚαλύτερο classical model: Random Forest")
        best = rf

    with open("models/classical_model.pkl", "wb") as f:
        pickle.dump(best, f)
    print("Αποθηκεύτηκε στο models/classical_model.pkl")

    return rf, xgb, results