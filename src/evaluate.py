import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from src.train_neural import DropoutNet

def evaluate_classical(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nRandom Forest")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Dropout","Dropout"]).plot(cmap="RdPu")
    plt.title("Confusion Matrix για Random Forest")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_rf.png")
    plt.show()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

def evaluate_neural(model, X_test, y_test):
    X_test_t = torch.FloatTensor(X_test)
    model.eval()
    with torch.no_grad():
        y_proba = model(X_test_t).numpy().flatten()

    y_pred = (y_proba >= 0.5).astype(int)

    print("\nNeural Network")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Dropout","Dropout"]).plot(cmap="RdPu")
    plt.title("Confusion Matrix για Neural Network")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_nn.png")
    plt.show()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

def compare_models(rf_metrics, nn_metrics):
    print("\nΣύγκριση Μοντέλων")
    print(f"{'Metric':<12} {'Random Forest':>15} {'Neural Network':>15}")
    print("-" * 45)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{metric:<12} {rf_metrics[metric]:>15.4f} {nn_metrics[metric]:>15.4f}")

    # Ποιο κερδίζει με βάση ROC-AUC
    if rf_metrics["f1"] >= nn_metrics["f1"]:
        print("\nΚαλύτερο μοντέλο: Random Forest")
        with open("models/classical_model.pkl", "rb") as f:
            best = pickle.load(f)
        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(best, f)
    else:
        print("\nΚαλύτερο μοντέλο: Neural Network")
        import shutil
        shutil.copy("models/neural_network.pt", "models/best_model.pt")
    print("Αποθηκεύτηκε ως best_model.")

def run_pca(X_train, y_train):
    print("\nPCA Analysis")
    pca_full = PCA()
    pca_full.fit(X_train)

    # Scree plot
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained)+1), explained, color="pink")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative)+1), cumulative, marker="o", color="pink")
    plt.axhline(y=0.9, color="red", linestyle="--", label="90%")
    plt.xlabel("Αριθμός Components")
    plt.ylabel("Cumulative Variance")
    plt.title("Cumulative Explained Variance")
    plt.legend()

    plt.tight_layout()
    plt.savefig("models/pca_scree.png")
    plt.show()

    n_90 = np.argmax(cumulative >= 0.9) + 1
    print(f"Components για 90% variance: {n_90}")

    # 2D scatter plot
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_train)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=y_train, cmap="coolwarm", alpha=0.4, s=10)
    plt.colorbar(scatter, label="Dropout")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D Projection")
    plt.tight_layout()
    plt.savefig("models/pca_2d.png")
    plt.show()

    # Top features στο PC1
    feature_importance_pc1 = np.abs(pca_2d.components_[0])
    top_idx = np.argsort(feature_importance_pc1)[::-1][:5]
    print(f"Top 5 features στο PC1 (indices): {top_idx}")