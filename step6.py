# Βήμα 6
import matplotlib.pyplot as plt
from step2 import data_parser
from step3 import extract_mfccs
from step5 import compute_feature_vectors, assign_colors_markers
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Step 2: Load data
wavs, speakers, digits = data_parser('./data/digits/')

# Step 3: Extract MFCCs
mfccs = extract_mfccs(wavs)

# Step 5: Compute feature vectors
feature_vectors, labels = compute_feature_vectors(mfccs)

'''
PCA Variance Retained
'''

def create_scaling_pca_pipeline(n_components=2):
    """
    Creates a scikit-learn Pipeline that scales the data and then applies PCA.

    Parameters:
    - n_components (int): Number of principal components to retain.

    Returns:
    - pipeline (sklearn.pipeline.Pipeline): The scaling and PCA pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])
    return pipeline

# Initialize PCA with 2 components
pca_2 = create_scaling_pca_pipeline(n_components=2)

# Fit PCA on the feature vectors and transform the data
principal_components_2 = pca_2.fit_transform(feature_vectors)

# Retrieve the percentage of variance explained by each principal component from the pipeline
variance_2 = pca_2.named_steps['pca'].explained_variance_ratio_

print(f"Variance Retained by the first two principal components: {variance_2.sum()*100:.2f}%")

'''
PCA 2D Plot
'''

# Assign colors and markers to each digit
unique_digits, digit_to_color, digit_to_marker = assign_colors_markers(labels)


# Initialize the plot
plt.figure(figsize=(12, 8))

for idx, (pc1, pc2) in enumerate(principal_components_2):
    digit = labels[idx]
    color = digit_to_color[digit]
    marker = digit_to_marker[digit]
    plt.scatter(pc1, pc2, color=color, marker=marker, s=100, edgecolors='k')

# Create custom legend to avoid duplicate labels
handles = []
for digit in unique_digits:
    handles.append(plt.Line2D([], [], color=digit_to_color[digit], marker=digit_to_marker[digit],
                              linestyle='', markersize=10, label=digit))

plt.legend(handles=handles, title='Digits')
plt.title('PCA Scatter Plot (2 Dimensions)')
plt.xlabel(f'Principal Component 1 ({variance_2[0]*100:.2f}% Variance)')
plt.ylabel(f'Principal Component 2 ({variance_2[1]*100:.2f}% Variance)')
plt.grid(True)
# Save the plot
plt.savefig('images/pca_2d_plot.png')
plt.show()

'''
PCA 3D Plot
'''

# Initialize PCA with 3 components
pca_3 = create_scaling_pca_pipeline(n_components=3)

# Fit PCA on the feature vectors and transform the data
principal_components_3 = pca_3.fit_transform(feature_vectors)

# Retrieve the percentage of variance explained by each principal component
variance_3 = pca_3.named_steps['pca'].explained_variance_ratio_

print(f"Variance Retained by the first three principal components: {variance_3.sum()*100:.2f}%")

# Plot the 3D scatter plot

# Initialize the plot
fig = plt.figure(figsize=(12, 8))

# Create 3D axes
ax = fig.add_subplot(111, projection='3d')

for idx, (pc1, pc2, pc3) in enumerate(principal_components_3):
    digit = labels[idx]
    color = digit_to_color[digit]
    marker = digit_to_marker[digit]
    ax.scatter(pc1, pc2, pc3, color=color, marker=marker, s=100, edgecolors='k')
    
# Create custom legend to avoid duplicate labels
handles = []
for digit in unique_digits:
    handles.append(plt.Line2D([], [], color=digit_to_color[digit], marker=digit_to_marker[digit],
                              linestyle='', markersize=10, label=digit))
    
ax.legend(handles=handles, title='Digits')
ax.set_title('PCA Scatter Plot (3 Dimensions)')
ax.set_xlabel(f'Principal Component 1 ({variance_3[0]*100:.2f}% Variance)')
ax.set_ylabel(f'Principal Component 2 ({variance_3[1]*100:.2f}% Variance)')
ax.set_zlabel(f'Principal Component 3 ({variance_3[2]*100:.2f}% Variance)')
# Save the plot
plt.savefig('images/pca_3d_plot.png')
plt.show()


