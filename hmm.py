import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from parser import parser
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 2  # the number of HMM states
n_mixtures = 2  # the number of Gaussians
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
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

    # TODO: YOUR CODE HERE
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val,
    ) = train_test_split(X, y, spk, test_size=0.2, random_state=42, stratify=y)
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        distributions = [Normal() for _ in range(n_mixtures)] # n_mixtures gaussian distributions
        a = GeneralMixtureModel(distributions, verbose=True).fit(
            np.concatenate(X)
        )  # Concatenate all frames from all samples into a large matrix
        dists.append(a)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        d = Normal()  # Fit a normal distribution on X
        d.fit(np.concatenate(X))
        dists.append(d)
    return dists


def initialize_transition_matrix(n_states):
    A = np.zeros((n_states, n_states), dtype=np.float32)
    for i in range(n_states):
        if i == n_states - 1:
            A[i, i] = 1.0
        else:
            A[i, i] = 0.5
            A[i, i + 1] = 0.5
    return A


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    p_start = np.zeros(n_states, dtype=np.float32)
    p_start[0] = 1.0
    return p_start


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    # Make sure the dtype is np.float32
    p_end = np.zeros(n_states, dtype=np.float32)
    p_end[-1] = 1.0
    return p_end


def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]

    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
        max_iter=100,
        tol=1e-3,
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  # create one hmm for each digit

    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        emission_model = (initialize_and_fit_gmm_distributions(X, n_states, n_mixtures) if gmm else
                          initialize_and_fit_normal_distributions(X, n_states))
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                # TODO: YOUR CODE HERE
                logp = hmm.log_probability(sample)
                ev[digit] = logp

            # TODO: YOUR CODE HERE
            predicted_digit = np.argmax(ev)  # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true


train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
hmms = train_hmms(train_dic, labels)


labels = list(set(y_train))
pred_val, true_val = evaluate(hmms, val_dic, labels)

pred_test, true_test = evaluate(hmms, test_dic, labels)


# TODO: YOUR CODE HERE
# Calculate and print the accuracy score on the validation and the test sets
# Plot the confusion matrix for the validation and the test set
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