import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Load MNIST dataset

print("Loading dataset...")
mnist = fetch_openml('mnist_784', version=1)

X = mnist.data.values / 255.0
y = mnist.target.astype(int).values.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Activation Functions

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Network Architecture

input_size = 784
hidden1 = 128
hidden2 = 64
output_size = 10


# Weight Initialization

np.random.seed(42)

W1 = np.random.randn(input_size, hidden1) * 0.01
b1 = np.zeros((1, hidden1))

W2 = np.random.randn(hidden1, hidden2) * 0.01
b2 = np.zeros((1, hidden2))

W3 = np.random.randn(hidden2, output_size) * 0.01
b3 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 30
batch_size = 64


# Training Loop (Mini-Batch)

for epoch in range(epochs):

    # Shuffle data
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices]
    y_train = y_train[indices]

    for i in range(0, X_train.shape[0], batch_size):

        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)

        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)

        Z3 = np.dot(A2, W3) + b3
        A3 = softmax(Z3)

        # Loss
        loss = -np.mean(np.sum(y_batch * np.log(A3 + 1e-8), axis=1))

        # Backprop
        m = X_batch.shape[0]

        dZ3 = A3 - y_batch
        dW3 = np.dot(A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")


# Testing

print("\nEvaluating model...")

Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2
A2 = relu(Z2)

Z3 = np.dot(A2, W3) + b3
A3 = softmax(Z3)

predictions = np.argmax(A3, axis=1)
true_labels = np.argmax(y_test, axis=1)

accuracy = np.mean(predictions == true_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save results
with open("output.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")

print("Results saved to output.txt")