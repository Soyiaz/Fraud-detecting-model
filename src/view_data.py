import numpy as np

# Load the data
X_train = np.load('data/processed/X_train_resampled.npz')

# Print the available keys
print("X_train keys:", X_train.files)

# Access the data using one of the keys
data = X_train['data']  # Adjust based on the actual key you want to access
print("Data shape:", data.shape)
print("Sample data:", data[:5])  # Print first 5 samples