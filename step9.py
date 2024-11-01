from sklearn.model_selection import train_test_split
from parser import parser

# You should have the following data after using parser:
X_train, X_test, y_train, y_test, spk_train, spk_test = parser("free-spoken-digit-dataset-1.0.10/recordings")

# Split X_train and y_train into a new X_train and X_val with an 80%-20% split, keeping stratification
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Confirm the split sizes and stratification
print(f"Training set size: {len(X_train)} samples")
print(f"Validation set size: {len(X_val)} samples")
