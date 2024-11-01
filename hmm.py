import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from parser import parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
n_states = 3  # Number of HMM states
n_mixtures = 2  # Number of Gaussians
gmm = True  # Use GMM or plain Gaussian
covariance_type = "diag"  # Diagonal covariance

def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic

def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("./free-spoken-digit-dataset-1.0.10/recordings", n_mfcc=13)
    X_train, X_val, y_train, y_val, spk_train, spk_val = train_test_split(X, y, spk, test_size=0.2, stratify=y, random_state=42)
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))
    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels

def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    dists = []
    X_concat = np.concatenate(X)
    X_concat = torch.tensor(X_concat, device=device)  # Ensure X_concat is on the correct device
    
    for _ in range(n_states):
        components = [Normal().to(device) for _ in range(n_mixtures)]
        gmm = GeneralMixtureModel(components, verbose=True).to(device)
        gmm.fit(X_concat)
        dists.append(gmm)
    return dists

def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        d = Normal().to(device)
        dists.append(d)
    return dists

def initialize_transition_matrix(n_states):
    A = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        if i < n_states - 1:
            A[i, i] = 0.7
            A[i, i + 1] = 0.3
        else:
            A[i, i] = 1.0
    return A

def initialize_starting_probabilities(n_states):
    start_probs = np.zeros(n_states, dtype=np.float32)
    start_probs[0] = 1.0
    return start_probs

def initialize_end_probabilities(n_states):
    end_probs = np.zeros(n_states, dtype=np.float32)
    end_probs[-1] = 1.0
    return end_probs

def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [torch.tensor(x, dtype=torch.float32, device=device) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=torch.tensor(A, device=device),
        starts=torch.tensor(start_probs, device=device),
        ends=torch.tensor(end_probs, device=device),
        verbose=True,
    ).to(device)
    model.fit(data)
    return model

def train_hmms(train_dic, labels):
    hmms = {}
    for dig in labels:
        X, _, _, _ = train_dic[dig]
        if gmm:
            emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
        else:
            emission_model = initialize_and_fit_normal_distributions(X, n_states)
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
    return hmms

def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            logp = []
            sample = torch.tensor(np.expand_dims(sample, 0), device=device, dtype=torch.float32)  # Ensure sample is on the correct device
            for digit in labels:
                hmm = hmms[digit]
                lp = hmm.log_probability(sample)
                logp.append(lp)
                
            predicted_digit = labels[np.argmax(torch.tensor(logp).cpu().numpy())]
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true

train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
hmms = train_hmms(train_dic, labels)

# Evaluation
pred_val, true_val = evaluate(hmms, val_dic, labels)
pred_test, true_test = evaluate(hmms, test_dic, labels)

# Calculate and print accuracy
val_accuracy = accuracy_score(true_val, pred_val)
test_accuracy = accuracy_score(true_test, pred_test)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot confusion matrices
val_conf_matrix = confusion_matrix(true_val, pred_val, labels=labels, normalize="true")
test_conf_matrix = confusion_matrix(true_test, pred_test, labels=labels, normalize="true")
plot_confusion_matrix(val_conf_matrix, classes=labels, title="Confusion Matrix - Validation Set", normalize=True)
plot_confusion_matrix(test_conf_matrix, classes=labels, title="Confusion Matrix - Test Set", normalize=True)
plt.show()
