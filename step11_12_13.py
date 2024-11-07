import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import product
import pandas as pd
from hmm2 import initialize_and_fit_gmm_distributions, train_single_hmm, evaluate, create_data, \
    initialize_and_fit_normal_distributions, train_hmms
import seaborn as sns
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix


def run_hmm_experiment(n_states, n_mixtures, train_dic, val_dic, test_dic, labels):
    """
    Run a single HMM experiment with given parameters
    """
    # Train HMMs with specified parameters
    hmms = {}
    for dig in labels:
        print(f"\nTraining HMM for digit {dig} with {n_states} states and {n_mixtures} mixtures")
        X, _, _, _ = train_dic[dig]

        try:
            if n_mixtures > 1:
                emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
            else:
                emission_model = initialize_and_fit_normal_distributions(X, n_states)

            hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
        except Exception as e:
            print(f"Error training HMM for digit {dig}: {e}")
            print("Trying with simpler model...")
            # Fallback to simpler model
            emission_model = initialize_and_fit_normal_distributions(X, n_states)
            hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)

    # Evaluate on validation and test sets
    pred_val, true_val = evaluate(hmms, val_dic, labels)
    pred_test, true_test = evaluate(hmms, test_dic, labels)

    val_accuracy = accuracy_score(true_val, pred_val)
    test_accuracy = accuracy_score(true_test, pred_test)

    return {
        'n_states': n_states,
        'n_mixtures': n_mixtures,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'pred_val': pred_val,
        'true_val': true_val,
        'pred_test': pred_test,
        'true_test': true_test
    }


def run_all_experiments(train_dic, val_dic, test_dic, labels):
    """
    Run experiments with different combinations of states and mixtures
    """
    states_range = range(2, 5)  # 1 to 4 states
    mixtures_range = range(1, 6)  # 1 to 5 Gaussian mixtures

    results = []
    for n_states, n_mixtures in product(states_range, mixtures_range):
        print(f"\nRunning experiment with {n_states} states and {n_mixtures} mixtures")
        result = run_hmm_experiment(n_states, n_mixtures, train_dic, val_dic, test_dic, labels)
        results.append(result)

    return results


def analyze_results(final_result, labels):
    """
    Analyze and visualize the final results
    """
    # Create confusion matrices
    val_cm = confusion_matrix(final_result['true_val'], final_result['pred_val'])
    test_cm = confusion_matrix(final_result['true_test'], final_result['pred_test'])

    # Plot normalized confusion matrices
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(val_cm, classes=labels, normalize=True,
                          title='Normalized Confusion Matrix - Validation Set')

    plt.subplot(1, 2, 2)
    plot_confusion_matrix(test_cm, classes=labels, normalize=True,
                          title='Normalized Confusion Matrix - Test Set')

    plt.tight_layout()
    plt.savefig('images/all_confusion_matrices.png')
    plt.show()


def plot_results(results):
    """
    Visualize the results of all experiments
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Create heatmap of validation accuracies
    plt.figure(figsize=(12, 8))
    pivot_val = df.pivot(index='n_states', columns='n_mixtures', values='val_accuracy')
    sns.heatmap(pivot_val, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Validation Accuracy by States and Mixtures')
    plt.xlabel('Number of Gaussian Mixtures')
    plt.ylabel('Number of HMM States')
    plt.show()

    # Create heatmap of test accuracies
    plt.figure(figsize=(12, 8))
    pivot_test = df.pivot(index='n_states', columns='n_mixtures', values='test_accuracy')
    sns.heatmap(pivot_test, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Test Accuracy by States and Mixtures')
    plt.xlabel('Number of Gaussian Mixtures')
    plt.ylabel('Number of HMM States')
    # Save the plot
    plt.savefig('heatmap.png')
    plt.show()

    # Find best configuration
    best_val_idx = df['val_accuracy'].idxmax()
    best_config = df.iloc[best_val_idx]
    print("\nBest configuration based on validation accuracy:")
    print(f"States: {best_config['n_states']}")
    print(f"Mixtures: {best_config['n_mixtures']}")
    print(f"Validation accuracy: {best_config['val_accuracy']:.3f}")
    print(f"Test accuracy: {best_config['test_accuracy']:.3f}")

    return best_config


def main():
    # Create data
    train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()

    # Run all experiments
    results = run_all_experiments(train_dic, val_dic, test_dic, labels)

    # Plot and analyze results
    best_config = plot_results(results)

    # Train final model with best configuration
    final_result = run_hmm_experiment(
        best_config['n_states'],
        best_config['n_mixtures'],
        train_dic, val_dic, test_dic, labels
    )

    # Analyze final results
    analyze_results(final_result, labels)

    return final_result, results


if __name__ == "__main__":
    final_result, all_results = main()
