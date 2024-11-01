from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from plot_confusion_matrix import plot_confusion_matrix

def evaluate_hmms(hmms, validation_dic, test_dic, labels):
    """
    Evaluates HMMs on the validation and test datasets.
    
    Parameters:
    - hmms: Dictionary of trained HMM models for each digit.
    - validation_dic: Dictionary containing validation data for each digit.
    - test_dic: Dictionary containing test data for each digit.
    - labels: List of digit labels.
    
    Returns:
    - Validation and test accuracies.
    """

    def predict_digit(hmms, sample):
        # Calculate log-likelihood for each model and return the model with max likelihood
        log_likelihoods = {digit: hmm.log_probability(sample) for digit, hmm in hmms.items()}
        predicted_digit = max(log_likelihoods, key=log_likelihoods.get)
        return predicted_digit

    # Validation Set Evaluation
    y_val_true, y_val_pred = [], []
    for digit, (samples, _, _, _) in validation_dic.items():
        for sample in samples:
            y_val_true.append(digit)
            y_val_pred.append(predict_digit(hmms, sample))

    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    print(f"Validation Accuracy: {val_accuracy:.2%}")
    val_conf_matrix = confusion_matrix(y_val_true, y_val_pred, labels=labels)
    print("Validation Confusion Matrix:")
    print(val_conf_matrix)

    # Test Set Evaluation
    y_test_true, y_test_pred = [], []
    for digit, (samples, _, _, _) in test_dic.items():
        for sample in samples:
            y_test_true.append(digit)
            y_test_pred.append(predict_digit(hmms, sample))

    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    test_conf_matrix = confusion_matrix(y_test_true, y_test_pred, labels=labels)
    print("Test Confusion Matrix:")
    print(test_conf_matrix)

    return val_accuracy, test_accuracy