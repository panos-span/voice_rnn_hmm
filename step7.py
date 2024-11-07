import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import librosa
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
# import one-hot encoder
from sklearn.preprocessing import OneHotEncoder

from step2 import data_parser
from step3 import extract_mfccs
from step5 import assign_colors_markers
from step5 import compute_feature_vectors
from sklearn.pipeline import Pipeline

# Step 2: Load data
wavs, speakers, digits = data_parser('./data/digits/')

# Step 3: Extract MFCCs
mfccs = extract_mfccs(wavs)

# Step 5: Compute feature vectors
feature_vectors, labels = compute_feature_vectors(mfccs)

unique_digits, digit_to_color, digit_to_marker = assign_colors_markers(labels)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3, random_state=42, stratify=labels)


'''
CustomBayesClassifier
'''

class CustomBayesClassifier:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.mean = {}
        self.var = {}
    
    def fit(self, X, y):
        """
        Fit the Bayesian classifier according to X, y.
        
        Parameters:
        - X (np.ndarray): Training feature vectors (num_samples, num_features).
        - y (list): Training labels.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            X_c = X[np.array(y) == cls]
            self.priors[cls] = X_c.shape[0] / X.shape[0]
            self.mean[cls] = np.mean(X_c, axis=0)
            self.var[cls] = np.var(X_c, axis=0) + 1e-6  # Add small value to prevent division by zero
    
    def _gaussian_log_prob(self, cls, x):
        """
        Calculate the log probability of x for class cls using Gaussian distribution.
        
        Parameters:
        - cls (str): Class label.
        - x (np.ndarray): Feature vector.
        
        Returns:
        - log_prob (float): Log probability.
        """
        mean = self.mean[cls]
        var = self.var[cls]
        # Calculate log Gaussian probability
        log_prob = -0.5 * np.sum(np.log(2. * np.pi * var))
        log_prob -= 0.5 * np.sum(((x - mean) ** 2) / var)
        return log_prob
    
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        
        Parameters:
        - X (np.ndarray): Test feature vectors (num_samples, num_features).
        
        Returns:
        - predictions (list): Predicted class labels.
        """
        predictions = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                # Compute log prior + log likelihood
                class_probs[cls] = np.log(self.priors[cls]) + self._gaussian_log_prob(cls, x)
            # Select the class with the highest probability
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return predictions
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        - X (np.ndarray): Test feature vectors (num_samples, num_features).
        - y (list): True class labels.
        
        Returns:
        - score (float): Mean accuracy.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


'''
Comparison with Gaussian Naive Bayes
'''

clfs = {
    'Custom Bayesian': CustomBayesClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42)
}

# Build pipelines for each classifier
pipelines = {}

for clf_name, clf in clfs.items():
    pipelines[clf_name] = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
# Keep resutls for each classifier in order to compare them
results = {}
    
# Train and evaluate each classifier
for clf_name, pipeline in pipelines.items():
    print(f"\n{clf_name} Classifier")
    # Train
    pipeline.fit(X_train, y_train)
    # Predict
    y_pred = pipeline.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Get f1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1 * 100:.2f}%")
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_digits)
    plot_confusion_matrix(conf_matrix, classes=unique_digits, title=f'Confusion Matrix - {clf_name}', normalize=True)
    # Classification report
    print(classification_report(y_test, y_pred, target_names=unique_digits))
    
    results[clf_name] = {'accuracy': accuracy, 'f1': f1}
    
# Compare classifiers based on accuracy and f1 score
print("\nComparison of Classifiers:")

# Get the best classifier based on accuracy
best_accuracy = max(results, key=lambda x: results[x]['accuracy'])
print(f"Best Classifier based on Accuracy: {best_accuracy}")

# Get the best classifier based on f1 score
best_f1 = max(results, key=lambda x: results[x]['f1'])
print(f"Best Classifier based on F1 Score: {best_f1}")

'''
Bonus
'''

def compute_feature_vectors_enhanced(mfccs, digits, wavs):
    """
    Processes audio files to extract enhanced feature vectors including MFCCs, deltas, delta-deltas,
    Zero-Crossing Rate (ZCR), and Spectral Centroid.
    
    Parameters:
    - mfccs (list): List of tuples, each containing MFCCs, deltas, and delta-deltas for an audio file.
    - digits (list): List of digit labels corresponding to each audio file.
    - wavs (list): List of waveform data for each audio file.
    
    Returns:
    - feature_vectors (np.ndarray): Array of enhanced feature vectors.
    - labels (list): List of digit labels corresponding to each feature vector.
    """
    
    feature_vectors = []
    labels = []
    
    for idx, ((mfcc, delta, delta2), wav) in enumerate(zip(mfccs, wavs)):
        # Concatenate MFCCs, Deltas, and Delta-Deltas
        combined_features = np.concatenate((mfcc, delta, delta2), axis=0)  # Shape: (39, frames)
        
        # Compute mean and standard deviation for each feature across all frames
        mean_features = np.mean(combined_features, axis=1)  # Shape: (39,)
        std_features = np.std(combined_features, axis=1)    # Shape: (39,)
        
        # Compute Zero-Crossing Rate (ZCR)
        zcr = librosa.feature.zero_crossing_rate(y=wav)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Compute Polynomial-Features
        poly = librosa.feature.poly_features(y=wav, sr=16000, hop_length=20, win_length=25, order=3)
        poly_mean = np.mean(poly)
        poly_std = np.std(poly)
        
        # Concatenate all features into a single vector
        feature_vector = np.concatenate((
            mean_features,       # 39
            std_features,        # 39
            [zcr_mean, zcr_std], # 2
            [poly_mean, poly_std] # 2
        ))  # Total length: 39 + 39 + 2 + 2 = 82
        
        feature_vectors.append(feature_vector)
        labels.append(digits[idx])
    
    feature_vectors = np.array(feature_vectors)
    return feature_vectors, labels


# Step 5: Compute feature vectors
feature_vectors, labels = compute_feature_vectors_enhanced(mfccs, digits, wavs)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.3, random_state=42, stratify=labels)

for clf_name, pipeline in pipelines.items():
    print(f"\n{clf_name} Classifier")
    # Train
    pipeline.fit(X_train, y_train)
    # Predict
    y_pred = pipeline.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # Get f1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1 * 100:.2f}%")
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_digits)
    plot_confusion_matrix(conf_matrix, classes=unique_digits, title=f'Confusion Matrix - {clf_name}', normalize=True)
    # Classification report
    print(classification_report(y_test, y_pred, target_names=unique_digits))
    
    results[clf_name] = {'accuracy': accuracy, 'f1': f1}
    
    
# Compare classifiers based on accuracy and f1 score
print("\nComparison of Classifiers Enchanced:")

# Get the best classifier based on accuracy
best_accuracy = max(results, key=lambda x: results[x]['accuracy'])
print(f"Best Classifier based on Accuracy: {best_accuracy}")

# Get the best classifier based on f1 score
best_f1 = max(results, key=lambda x: results[x]['f1'])
print(f"Best Classifier based on F1 Score: {best_f1}")
