import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from itertools import product
import pandas as pd
from hmm import create_data, initialize_and_fit_gmm_distributions, initialize_and_fit_normal_distributions, train_single_hmm, evaluate
import seaborn as sns
import numpy as np

def add_noise_to_data(X, noise_level=1e-6):
    """Add small amount of noise to prevent singular matrices"""
    return [x + np.random.normal(0, noise_level, x.shape) for x in X]


def run_hmm_experiment(n_states:int, n_mixtures:int, train_dic:dict, val_dic:dict, test_dic:dict, labels:list) -> dict:
    """
    Run a single HMM experiment with the specified parameters.
    
    Parameters:
    - n_states (int): Number of HMM states.
    - n_mixtures (int): Number of Gaussians.
    - train_dic (dict): Dictionary containing training data.
    - val_dic (dict): Dictionary containing validation data.
    - test_dic (dict): Dictionary containing test data.
    - labels (list): List of labels.
    """
    hmms = {}
    for digit in labels:
        print(f"Training HMM for digit {digit} with {n_states} states and {n_mixtures} mixtures")
        X, _, _ , _ = train_dic[digit]
        try:
            if n_mixtures > 1:
                emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
            else:
                emission_model = initialize_and_fit_normal_distributions(X, n_states)
            hmms[digit] = train_single_hmm(X, emission_model, digit, n_states)
        except Exception as e:
            print(f"Error training HMM for digit {digit}: {e}")
            print("Trying with 1 mixture")
            emission_model = initialize_and_fit_normal_distributions(X, n_states)
            hmms[digit] = train_single_hmm(X, emission_model, digit, n_states)
    
    # Evaluate on validation set
    pred_val , true_val = evaluate(hmms, val_dic, labels)
    pred_test, true_test = evaluate(hmms, test_dic, labels)
    
    val_acc = accuracy_score(true_val, pred_val)
    test_acc = accuracy_score(true_test, pred_test)
    
    print(f"Validation accuracy: {val_acc}")
    print(f"Test accuracy: {test_acc}")
    
    return {
        'n_states': n_states,
        'n_mixtures': n_mixtures,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'pred_val': pred_val,
        'true_val': true_val,
        'pred_test': pred_test,
        'true_test': true_test
    }
    
def run_all_hmm_experiments(train_dic:dict, val_dic:dict, test_dic:dict, labels:list) -> pd.DataFrame:
    """
    Run all HMM experiments with different values of n_states and n_mixtures.
    
    Parameters:
    - train_dic (dict): Dictionary containing training data.
    - val_dic (dict): Dictionary containing validation data.
    - test_dic (dict): Dictionary containing test data.
    - labels (list): List of labels.
    """
    states_range = range(2, 5)
    mixtures_range = range(1, 6)
    
    results = []
    for n_states, n_mixtures in product(states_range, mixtures_range):
        print(f"Running experiment with {n_states} states and {n_mixtures} mixtures")
        results.append(run_hmm_experiment(n_states, n_mixtures, train_dic, val_dic, test_dic, labels))
    return pd.DataFrame(results)

def analyze_resutls(final_result: pd.DataFrame, labels: list) -> None:
    """
    Analyze the results of the HMM experiments.
    
    Parameters:
    - final_result (pd.DataFrame): DataFrame containing the results of the experiments.
    - labels (list): List of labels.
    """
    # Create the confusion matrices
    val_cm = confusion_matrix(final_result['true_val'], final_result['pred_val'], labels=labels)
    test_cm = confusion_matrix(final_result['true_test'], final_result['pred_test'], labels=labels)
    
    # Plot the confusion matrices
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(val_cm, labels, title="Validation Confusion Matrix", normalize=True)
    plt.subplot(1, 2, 2)
    
    plot_confusion_matrix(test_cm, labels, title="Test Confusion Matrix", normalize=True)
    plt.tight_layout()
    plt.savefig("./images/hmm_confusion_matrices.png")
    plt.show()
    
def plot_heatmaps(results: pd.DataFrame, metric: str) -> None:
    """
    Plot the heatmaps of the HMM experiments.
    
    Parameters:
    - results (pd.DataFrame): DataFrame containing the results of the experiments.
    - metric (str): Metric to plot ('val_acc' or 'test_acc').
    """
    plt.figure(figsize=(15, 5))
    pivot = results.pivot(index='n_states', columns='n_mixtures', values=metric)
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt=".3f", cbar_kws={'label': metric})
    plt.title(f"{metric} Heatmap by Number of States and Mixtures")
    plt.xlabel("Number of Mixtures")
    plt.ylabel("Number of States")
    plt.savefig(f"./images/hmm_{metric}_heatmap.png")
    plt.show()
    
    
    
def plot_resutls(results: pd.DataFrame) -> dict:
    """
    Plot the results of the HMM experiments.
    
    Parameters:
    - results (pd.DataFrame): DataFrame containing the results of the experiments.
    """
    # Plot the heatmaps
    plot_heatmaps(results, 'val_acc')
    plot_heatmaps(results, 'test_acc')
    
    # Find the best configuration
    best_val_idx = results.loc['val_acc'].idxmax()
    best_config = results.loc[best_val_idx]
    print(f"Best configuration: {best_config['n_states']} states, {best_config['n_mixtures']} mixtures")
    print(f"Validation accuracy: {best_config['val_acc']}")
    print(f"Test accuracy: {best_config['test_acc']}")
    
    return best_config

def main():
    # Load the data
    train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
    
    # Run the experiments
    results = run_all_hmm_experiments(train_dic, val_dic, test_dic, labels)
    
    # Plot and analyze the results
    best_config = plot_resutls(results)
    
    # Train the best model
    final_result = run_hmm_experiment(best_config['n_states'], best_config['n_mixtures'], train_dic, val_dic, test_dic, labels)
    
    # Analyze the results
    analyze_resutls(final_result, labels)
    
    return final_result, results

if __name__ == "__main__":
    final_result, results = main()
    print(final_result)
    