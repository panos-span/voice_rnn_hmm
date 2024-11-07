import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from itertools import product
import pandas as pd
from hmm import (
    create_data,
    initialize_and_fit_gmm_distributions,
    initialize_and_fit_normal_distributions,
    train_single_hmm,
    evaluate,
)
import seaborn as sns


def run_hmm_experiment(
    n_states: int, n_mixtures: int, train_dic: dict, val_dic: dict, test_dic: dict, labels: list
) -> dict:
    """
    Run a single HMM experiment with the specified parameters.
    """
    hmms = {}
    for digit in labels:
        print(f"Training HMM for digit {digit} with {n_states} states and {n_mixtures} mixtures")
        X, _, _, _ = train_dic[digit]
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

    # Evaluate on validation and test sets
    pred_val, true_val = evaluate(hmms, val_dic, labels)
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
        'true_test': true_test,
    }


def run_all_hmm_experiments(train_dic: dict, val_dic: dict, test_dic: dict, labels: list) -> pd.DataFrame:
    """
    Run all HMM experiments with different values of n_states and n_mixtures.
    """
    states_range = range(1, 5)
    mixtures_range = range(1, 6)

    results = []
    for n_states, n_mixtures in product(states_range, mixtures_range):
        print(f"Running experiment with {n_states} states and {n_mixtures} mixtures")
        experiment_result = run_hmm_experiment(n_states, n_mixtures, train_dic, val_dic, test_dic, labels)
        results.append(experiment_result)
    return pd.DataFrame(results)


def analyze_results(final_result: dict, labels: list) -> None:
    """
    Analyze the results of the HMM experiments.
    
    Parameters:
    - final_result (dict): Dictionary containing prediction and true values for validation and test sets
    - labels (list): List of digit labels (0-9)
    """
    # Convert labels to list if they aren't already
    labels = list(map(int, labels))  # Ensure labels are integers
    labels.sort()  # Sort labels 0-9
    
    # Create the confusion matrices
    val_cm = confusion_matrix(final_result['true_val'], final_result['pred_val'], labels=labels)
    test_cm = confusion_matrix(final_result['true_test'], final_result['pred_test'], labels=labels)
    
    # Create figure with two subplots
    plt.figure(figsize=(20, 8))
    
    # Plot validation confusion matrix
    plot_confusion_matrix(val_cm, 
                         classes=labels,
                         normalize=True,
                         title='Normalized Validation Confusion Matrix')
    
    # Plot test confusion matrix
    plot_confusion_matrix(test_cm, 
                         classes=labels,
                         normalize=True,
                         title='Normalized Test Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig("./images/hmm_confusion_matrices.png", bbox_inches='tight', dpi=300)
    plt.show()      
    # Print accuracy scores
    val_acc = accuracy_score(final_result['true_val'], final_result['pred_val'])
    test_acc = accuracy_score(final_result['true_test'], final_result['pred_test'])
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")


def plot_heatmaps(results: pd.DataFrame, metric: str) -> None:
    """
    Plot the heatmaps of the HMM experiments.
    """
    plt.figure(figsize=(15, 5))
    pivot = results.pivot(index='n_states', columns='n_mixtures', values=metric)
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt=".3f", cbar_kws={'label': metric})
    plt.title(f"{metric} Heatmap by Number of States and Mixtures")
    plt.xlabel("Number of Mixtures")
    plt.ylabel("Number of States")
    plt.savefig(f"./images/hmm_{metric}_heatmap.png")
    plt.show()


def plot_results(results: pd.DataFrame) -> dict:
    """
    Plot the results of the HMM experiments and find the best configuration.
    """
    # Plot the heatmaps
    plot_heatmaps(results, 'val_acc')
    plot_heatmaps(results, 'test_acc')

    # Find the best configuration based on validation accuracy
    best_val_idx = results['val_acc'].idxmax()
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
    best_config = plot_results(results)

    # Train the best model
    final_result = run_hmm_experiment(
        int(best_config['n_states']), int(best_config['n_mixtures']), train_dic, val_dic, test_dic, labels
    )

    # Analyze the results
    analyze_results(final_result, labels)

    return final_result, results


if __name__ == "__main__":
    final_result, results = main()
    print(final_result)
