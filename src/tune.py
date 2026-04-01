import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score

def tune_classical(X_train, y_train, X_val, y_val):
    print("Hyperparameter Tuning Random Forest")

    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None]
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"\nΚαλύτερες παράμετροι: {search.best_params_}")
    print(f"Καλύτερο CV F1: {search.best_score_:.4f}")

    # Αξιολόγηση στο validation set
    best_model = search.best_estimator_
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"Validation F1: {f1_score(y_val, y_val_pred):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")

    # Αποθήκευση
    with open("models/classical_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Αποθηκεύτηκε.")

    return best_model