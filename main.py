from src.preprocessing import preprocess
from src.train_classical import train_classical
from src.train_neural import train_neural, DropoutNet
from src.tune import tune_classical
from src.evaluate import evaluate_classical, evaluate_neural, compare_models, run_pca
import pickle
import torch

def main():
    print("=" * 50)
    print("1st Assignment: Student Dropout Prediction")
    print("=" * 50)

    # Task 2 - Preprocessing
    print("\nTask 2: Preprocessing")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess()

    # Task 2.7 - PCA
    print("\nPCA Analysis")
    run_pca(X_train, y_train)

    # Task 3.1 - Classical Models
    print("\nTask 3.1: Classical Models")
    rf, xgb, results = train_classical(X_train, y_train, X_val, y_val, feature_names)

    # Task 3.2 - Neural Network
    print("\nTask 3.2: Neural Network")
    nn_model, train_losses, val_losses = train_neural(X_train, y_train, X_val, y_val)

    # Task 4 - Evaluation
    print("\nTask 4: Evaluation")
    with open("models/classical_model.pkl", "rb") as f:
        best_classical = pickle.load(f)

    rf_metrics = evaluate_classical(best_classical, X_test, y_test)

    input_size = X_train.shape[1]
    nn = DropoutNet(input_size)
    nn.load_state_dict(torch.load("models/neural_network.pt"))
    nn_metrics = evaluate_neural(nn, X_test, y_test)

    compare_models(rf_metrics, nn_metrics)

    # Task 6 - Hyperparameter Tuning
    print("\nTask 6: Hyperparameter Tuning")
    tune_classical(X_train, y_train, X_val, y_val)

    print("\nΤο Pipeline έχει ολοκληρωθεί.")

if __name__ == "__main__":
    main()