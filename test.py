import pickle
import numpy as np
import argparse
from utils.utils import custom_softmax, mapping_three_prob_to_class, accuracy
from utils.config import CONFIG

# Command-line argument support
parser = argparse.ArgumentParser(description="Test Model with Different Datasets")
parser.add_argument("--data", type=str, default=CONFIG["data"]["test_data"], help="Path to test dataset")
args = parser.parse_args()

# Load test data (default or provided via CLI)
test_data_path = args.data
with open(test_data_path, "rb") as f:
    test_data = pickle.load(f)

X_test, y_test = test_data["X"], test_data["y"]

# Load trained model
model_path = CONFIG["model"]["saved_model"]
with open(model_path, "rb") as f:
    trained_weights = pickle.load(f)

# Predict
y_logistic_test = (trained_weights.T @ X_test.T).T
y_logistic_test = custom_softmax(y_logistic_test, axis=1)
y_pred = mapping_three_prob_to_class(y_logistic_test)

# Compute accuracy
acc_test = accuracy(y_pred, y_test)
print(f" Model Testing Complete! Accuracy: {acc_test:.2f}%")
