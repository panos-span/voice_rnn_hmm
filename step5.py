'''
Μια πρώτη προσέγγιση για την αναγνώριση των ψηφίων είναι η εξαγωγή ενός μοναδικού διανύσματος
χαρακτηριστικών για κάθε εκφώνηση.
Ενώστε τα mfccs – deltas – delta-deltas και έπειτα για κάθε εκφώνηση δημιουργείστε ένα διάνυσμα παίρνοντας
τη μέση τιμή και την τυπική απόκλιση κάθε χαρακτηριστικού για όλα τα παράθυρα της εκφώνησης.
Αναπαραστήστε με scatter plot τις 2 πρώτες διαστάσεις των διανυσμάτων αυτών, χρησιμοποιώντας διαφορετικό
χρώμα και σύμβολο για κάθε ψηφίο. Σχολιάστε το διάγραμμα.
'''

import librosa
import numpy as np
import matplotlib.pyplot as plt
from step3 import extract_mfccs
from step2 import data_parser

# Step 2: Load data
wavs, speakers, digits = data_parser('./data/digits/')

# Step 3: Extract MFCCs
mfccs = extract_mfccs(wavs)

'''
Feature Extraction
'''
def compute_feature_vectors(mfccs):
    """
    Processes audio files in the specified directory to extract feature vectors.

    Parameters:
    - mfccs (list): List of tuples, each containing MFCCs, deltas, and delta-deltas for an audio file.

    Returns:
    - feature_vectors (np.ndarray): Array of feature vectors (shape: [num_utterances, 78]).
    - labels (list): List of digit labels corresponding to each feature vector.
    """
    
    # Step 3: Initialize lists to store feature vectors and corresponding labels
    feature_vectors = []
    labels = []
    
    for idx, (mfcc, delta, delta2) in enumerate(mfccs):
        # Concatenate MFCCs, Deltas, and Delta-Deltas along the feature axis
        combined_features = np.concatenate((mfcc, delta, delta2), axis=0)  # Shape: (39, frames)
        
        # Compute mean and standard deviation for each feature across all frames
        mean_features = np.mean(combined_features, axis=1)  # Shape: (39,)
        std_features = np.std(combined_features, axis=1)    # Shape: (39,)
        
        # Concatenate mean and std to form a single feature vector (total length: 78)
        feature_vector = np.concatenate((mean_features, std_features))  # Shape: (78,)
        
        # Append to the list of feature vectors
        feature_vectors.append(feature_vector)
        
        # Append the corresponding digit label
        labels.append(digits[idx])
    
    # Convert the list of feature vectors to a NumPy array for easier manipulation
    feature_vectors = np.array(feature_vectors)  # Shape: (num_utterances, 78)
    
    return feature_vectors, labels


feature_vectors, labels = compute_feature_vectors(mfccs)

'''
Prepare for visualization
'''

def assign_colors_markers(labels):
    """
    Identifies unique digits and assigns unique colors and markers to each digit.
    
    Parameters:
    - labels (list): List of digit labels corresponding to each data point.
    
    Returns:
    - unique_digits (list): Sorted list of unique digit labels.
    - digit_to_color (dict): Mapping from each digit to a unique color.
    - digit_to_marker (dict): Mapping from each digit to a unique marker.
    """
    # Identify unique digits in the dataset
    unique_digits = sorted(set(labels))
    
    # Assign unique colors and markers to each digit
    colors = plt.cm.get_cmap('tab10', len(unique_digits))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # Extend or modify as needed
    
    digit_to_color = {digit: colors(idx) for idx, digit in enumerate(unique_digits)}
    digit_to_marker = {digit: markers[idx % len(markers)] for idx, digit in enumerate(unique_digits)}
    
    return unique_digits, digit_to_color, digit_to_marker


unique_digits, digit_to_color, digit_to_marker = assign_colors_markers(labels)

'''
Scatterplot of feature vectors
'''

# Initialize the plot
plt.figure(figsize=(12, 8))

for idx, feature_vector in enumerate(feature_vectors):
    digit = labels[idx]
    color = digit_to_color[digit]
    marker = digit_to_marker[digit]
    
    # Extract the first two dimensions (mean of the first two features)
    x = feature_vector[0]  # Mean of first feature
    y = feature_vector[1]  # Mean of second feature
    
    # Plot the point
    plt.scatter(x, y, color=color, marker=marker, label=digit, edgecolors='k', s=100)

# Create custom legend to avoid duplicate labels
handles = []
for digit in unique_digits:
    handles.append(plt.Line2D([], [], color=digit_to_color[digit], marker=digit_to_marker[digit],
                              linestyle='', markersize=10, label=digit))

plt.legend(handles=handles, title='Digits')
plt.title('Scatter Plot of Feature Vectors (First Two Dimensions)')
plt.xlabel('Mean of Feature 1')
plt.ylabel('Mean of Feature 2')
plt.grid(True)
plt.show()


