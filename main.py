import numpy as np
import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles

from utils import test_accuracy, plot_decision_boundary


# Sigmoid and ReLU activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# Feedforward Neural Network class
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def loss(self, y, y_pred):
        # Mean Squared Error loss
        return np.mean((y - y_pred) ** 2)
        # Binary cross-entropy loss
        # return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def forward_propagation_gradients(self, X, y):
        # Compute gradients using forward propagation (less efficient)
        pass  # Placeholder for manual gradient calculation

    def backpropagation_gradients(self, X, y):
        # Backpropagation (efficient)
        m = X.shape[0]

        # Forward pass
        y_pred = self.forward(X)

        # Backward pass

        # Derivative of the loss function with respect to z2, z2 is the input to the activation function of the output layer
        # This will tell us how to change z2 to reduce the loss
        # dL/dz2
        dz2 = y_pred - y

        # dW2 is the derivative of the loss function with respect to W2
        # dL/dW2 = dL/dz2 * dz2/dW2
        #        = dz2 * d(a1*W2 + b2)/dW2
        #        = dz2 * a1
        # then we average over all samples
        dW2 = (1 / m) * np.dot(self.a1.T, dz2)
        # db2 is the derivative of the loss function with respect to b2
        # dL/db2 = dL/dz2 * dz2/db2
        #        = dz2 * d(a1*W2 + b2)/db2
        #        = dz2 * 1
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        dz1 = np.dot(dz2, self.W2.T) * (self.a1 > 0)
        dW1 = (1 / m) * np.dot(X.T, dz1)
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_weights(self, gradients, lr=0.01):
        # Update weights using gradients
        dW1, db1, dW2, db2 = gradients
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def visualize_training_data(X_train, y_train):
    # visualize the training data:
    import matplotlib.pyplot as plt
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='class 0')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Training Data')
    plt.legend()
    plt.show()





np.random.seed(3)

# Example usage
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# breaks otherwise
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = SimpleNN(input_size=2, hidden_size=3, output_size=1)

# Train using backpropagation
start_time = time.time()
# for epoch in range(30000):
accuracy = 0
epoch = 0
max_epochs = 100000
while accuracy < 0.99 and epoch < max_epochs:
    if epoch % 2000 == 0:
        y_pred_test = nn.forward(X_test)
        print(f"Epoch {epoch}")
        accuracy = test_accuracy(y_test, y_pred_test)
        print(f"Test accuracy: {accuracy:.2f}")
        loss = nn.loss(y_test, y_pred_test)
        print(f"Test loss: {loss:.2f}")

        plot_decision_boundary(X_test, y_test, nn)
        time.sleep(1)
        print()

    gradients = nn.backpropagation_gradients(X_train, y_train)
    nn.update_weights(gradients)
    epoch += 1

if epoch == max_epochs:
    print("Failed to converge")

backprop_time = time.time() - start_time
print(f"Training with backpropagation took {backprop_time:.2f} seconds")


# Plot decision boundary
plot_decision_boundary(X_test, y_test, nn)
