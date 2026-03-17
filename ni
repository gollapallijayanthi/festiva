import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import imdb

# Build Neural Network
model = models.Sequential([
    layers.Dense(16, activation='relu', input_shape=(10000,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# Combine train and test data
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))

# Dataset information
print("Training data:")
print(X.shape)
print(y.shape)

print("Classes:")
print(np.unique(y))

print("Number of unique words:")
print(len(np.unique(np.hstack(X))))

# Review length statistics
lengths = [len(i) for i in X]
print("Review length:")
print("Mean %.2f words (Std: %.2f)" % (np.mean(lengths), np.std(lengths)))

# Plot review length distribution
plt.boxplot(lengths)
plt.title("Review Length Distribution")
plt.ylabel("Number of Words")
plt.show()
