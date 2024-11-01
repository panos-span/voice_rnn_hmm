import numpy as np
from pomegranate.hmm import DenseHMM
from hmm import initialize_and_fit_gmm_distributions, initialize_transition_matrix, initialize_starting_probabilities, initialize_end_probabilities, create_data

def train_hmms_with_em(train_dic, labels, max_states=4, max_mixtures=5, 
                       max_iterations=100, tolerance=1e-3):
    """
    Trains HMM models for each digit using EM, iterating over various HMM states and Gaussian mixtures.
    
    Parameters:
    - train_dic: Dictionary containing training data by digit.
    - labels: List of digit labels.
    - max_states: Maximum number of states for each HMM (1 to max_states).
    - max_mixtures: Maximum number of Gaussian mixtures per state (1 to max_mixtures).
    - max_iterations: Maximum iterations for EM.
    - tolerance: Convergence threshold for log-likelihood change.
    
    Returns:
    - trained_models: Dictionary of trained HMM models for each digit.
    """
    trained_models = {}

    for digit in labels:
        X, _, _, _ = train_dic[digit]
        
        best_model = None
        best_log_likelihood = -np.inf

        # Try different configurations of states and mixtures
        for n_states in range(1, max_states + 1):
            for n_mixtures in range(1, max_mixtures + 1):
                
                # Initialize GMM for each state
                emissions = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)

                # Initialize transition matrix and starting probabilities
                A = initialize_transition_matrix(n_states)
                start_probs = initialize_starting_probabilities(n_states)
                end_probs = initialize_end_probabilities(n_states)
                
                # Fit the HMM model using EM
                model = DenseHMM(
                    distributions=emissions,
                    edges=A,
                    starts=start_probs,
                    ends=end_probs,
                    verbose=True,
                )

                # Train with EM and monitor log likelihood for convergence
                prev_log_likelihood = -np.inf
                for i in range(max_iterations):
                    log_likelihood = model.fit(X, max_iterations=1)
                    if abs(log_likelihood - prev_log_likelihood) < tolerance:
                        print(f"Converged at iteration {i + 1} for digit {digit}" 
                              f"with {n_states} states and {n_mixtures} mixtures.")
                        break
                    prev_log_likelihood = log_likelihood

                # Update the best model if log-likelihood is improved
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_model = model

        # Save the best model for the digit
        trained_models[digit] = best_model
        print(f"Trained best model for digit {digit} with log-likelihood: {best_log_likelihood}")

    return trained_models

train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
trained_models = train_hmms_with_em(train_dic, labels, max_states=4, max_mixtures=5, max_iterations=100, tolerance=1e-3)
