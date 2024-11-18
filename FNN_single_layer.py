import time

import numpy as np
import sympy
from sklearn.model_selection import train_test_split
from sympy import symbols, diff

from circuits import Variable, Constant, Add, Multiply, Expression, Exponent, forward_propagation_partial, Sigmoid, \
    back_propagation_partial
from utils import test_accuracy, plot_decision_boundary


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


class SimpleNN:
    def __init__(self, input_size, output_size):
        """
        Initialize weights
        :param input_size:
        :param output_size:
        """
        self.W1 = np.random.randn(input_size, output_size)
        self.b1 = np.zeros((1, output_size))

        self.z1 = None
        self.a1 = None

    def forward(self, X):
        """
        Forward pass
        :param X:
        :return:
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        return self.a1

    @staticmethod
    def loss(y, y_pred):
        """
        Calculates loss function value for given true and predicted values
        :param y:
        :param y_pred:
        :return:
        """
        # Mean Squared Error loss
        return np.mean((y - y_pred) ** 2)
        # Binary cross-entropy loss
        # return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def backpropagation_gradients(self, X, y):
        """
        Calculates gradients using backpropagation (typical approach)
        :param X:
        :param y:
        :return:
        """
        m = X.shape[0]

        y_pred = self.forward(X)

        dz1 = y_pred - y  # dL/dz1

        dW1 = np.dot(X.T, dz1) / m  # dL/dW1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # dL/db1

        return dW1, db1

    def backpropagation_with_actual_gradients(self, X, y):
        """
        Backpropagation with gradients calculated with sympy
        :param X:
        :param y:
        :return:
        """
        start1 = time.time()
        s_W1_1, s_W1_2, s_b1, s_X1, s_X2, s_y = symbols('W1_1 W1_2 b1 X1 X2 y')

        s_z1 = s_X1 * s_W1_1 + s_X2 * s_W1_2 + s_b1
        s_a1 = 1 / (1 + sympy.exp(-s_z1))  # sigmoid activation
        # s_a1 = s_z1  # linear activation

        s_loss = (s_a1 - s_y) ** 2  # Mean Squared Error loss for one sample

        # Derivatives of the loss w.r.t weights and bias
        s_dloss_dW1_1 = diff(s_loss, s_W1_1)
        s_dloss_dW1_2 = diff(s_loss, s_W1_2)
        s_dloss_db1 = diff(s_loss, s_b1)
        end1 = time.time()
        # print(f"Symbolic differentiation took {end1 - start1:.2f} seconds")

        start2 = time.time()
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        for i in range(X.shape[0]):
            X1_val, X2_val = X[i]
            y_val = y[i, 0]

            grad_W1_1 = s_dloss_dW1_1.subs(
                {s_W1_1: self.W1[0, 0], s_W1_2: self.W1[1, 0], s_b1: self.b1[0, 0], s_X1: X1_val, s_X2: X2_val,
                 s_y: y_val})
            grad_W1_2 = s_dloss_dW1_2.subs(
                {s_W1_1: self.W1[0, 0], s_W1_2: self.W1[1, 0], s_b1: self.b1[0, 0], s_X1: X1_val, s_X2: X2_val,
                 s_y: y_val})
            grad_b1 = s_dloss_db1.subs(
                {s_W1_1: self.W1[0, 0], s_W1_2: self.W1[1, 0], s_b1: self.b1[0, 0], s_X1: X1_val, s_X2: X2_val,
                 s_y: y_val})

            dW1[0, 0] += grad_W1_1
            dW1[1, 0] += grad_W1_2
            db1[0, 0] += grad_b1

        dW1 /= X.shape[0]
        db1 /= X.shape[0]
        end2 = time.time()
        # print(f"Actual differentiation took {end2 - start2:.2f} seconds")

        return dW1, db1

    def backpropagation_with_forward_AD(self, X, y):
        """
        Backpropagation with gradients calculated with forward accumulation
        :param X:
        :param y:
        :return:
        """
        s_X1 = Variable('X1')
        s_X2 = Variable('X2')
        s_W1_1 = Variable('W1_1')
        s_W1_2 = Variable('W1_2')
        s_b1 = Variable('b1')
        s_y = Variable('y')

        s_z1 = Add(Add(Multiply(s_X1, s_W1_1), Multiply(s_X2, s_W1_2)), s_b1)
        s_sigmoid_loss = Expression(Exponent(Add(Sigmoid(s_z1), Multiply(Constant(-1), s_y)), Constant(2)))
        partial_W1_1 = forward_propagation_partial(s_sigmoid_loss, s_W1_1)
        partial_W1_2 = forward_propagation_partial(s_sigmoid_loss, s_W1_2)
        partial_b1 = forward_propagation_partial(s_sigmoid_loss, s_b1)

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        for i in range(X.shape[0]):
            X1_val, X2_val = X[i]
            y_val = y[i, 0]
            partial_W1_1_val = partial_W1_1.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})
            partial_W1_2_val = partial_W1_2.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})
            partial_b1_val = partial_b1.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})

            dW1[0, 0] += partial_W1_1_val
            dW1[1, 0] += partial_W1_2_val
            db1[0, 0] += partial_b1_val

        dW1 /= X.shape[0]
        db1 /= X.shape[0]

        return dW1, db1

    def backpropagation_with_reverse_AD(self, X, y):
        """
        Backpropagation with gradients calculated with reverse accumulation
        :param X:
        :param y:
        :return:
        """
        s_X1 = Variable('X1')
        s_X2 = Variable('X2')
        s_W1_1 = Variable('W1_1')
        s_W1_2 = Variable('W1_2')
        s_b1 = Variable('b1')
        s_y = Variable('y')

        s_z1 = Add(Add(Multiply(s_X1, s_W1_1), Multiply(s_X2, s_W1_2)), s_b1)
        s_sigmoid_loss = Expression(Exponent(Add(Sigmoid(s_z1), Multiply(Constant(-1), s_y)), Constant(2)))
        partials = back_propagation_partial(s_sigmoid_loss)
        partial_W1_1 = partials.get_derivative(s_W1_1)
        partial_W1_2 = partials.get_derivative(s_W1_2)
        partial_b1 = partials.get_derivative(s_b1)

        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        for i in range(X.shape[0]):
            X1_val, X2_val = X[i]
            y_val = y[i, 0]
            partial_W1_1_val = partial_W1_1.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})
            partial_W1_2_val = partial_W1_2.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})
            partial_b1_val = partial_b1.evaluate(
                {'X1': X1_val, 'X2': X2_val, 'W1_1': self.W1[0, 0], 'W1_2': self.W1[1, 0], 'b1': self.b1[0, 0],
                 'y': y_val})

            dW1[0, 0] += partial_W1_1_val
            dW1[1, 0] += partial_W1_2_val
            db1[0, 0] += partial_b1_val

        dW1 /= X.shape[0]
        db1 /= X.shape[0]

        return dW1, db1


    def update_weights(self, gradients, lr=0.01):
        """
        Update weights using gradients
        :param gradients:
        :param lr:
        :return:
        """
        dW1, db1 = gradients
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


np.random.seed(3)

# Generate random x values for two classes
x_class1 = np.random.rand(100, 2) * [1, 1] + [0, 1]  # Points above the line
x_class2 = np.random.rand(100, 2) * [1, 1] + [1, -1]  # Points below the line

# Combine data and create labels
X = np.vstack((x_class1, x_class2))
y = np.hstack((np.ones(100), np.zeros(100)))  # 1 for class above line, 0 for below
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = SimpleNN(2, 1)

gradients3_total = 0
gradients4_total = 0

start_time = time.time()
accuracy = 0
epoch = 0
max_epochs = 10000
while accuracy < 0.99 and epoch < max_epochs:
    if epoch % 100 == 0:
        y_pred_test = nn.forward(X_test)
        print(f"Epoch {epoch}")
        accuracy = test_accuracy(y_test, y_pred_test)
        print(f"Test accuracy: {accuracy:.2f}")
        loss = SimpleNN.loss(y_test, y_pred_test)
        print(f"Test loss: {loss:.2f}")

        plot_decision_boundary(X_test, y_test, nn)
        time.sleep(0.5)
        print()

    # gradients = nn.backpropagation_gradients(X_train, y_train)
    # print("gradients", gradients[0][0], gradients[0][1], gradients[1])

    # gradients2 = nn.backpropagation_with_actual_gradients(X_train, y_train)
    # print("gradients2", gradients2[0][0], gradients2[0][1], gradients2[1])

    start = time.time()
    gradients3 = nn.backpropagation_with_forward_AD(X_train, y_train)
    end = time.time() - start
    gradients3_total += end
    # print("gradients3", gradients3[0][0], gradients3[0][1], gradients3[1])

    start = time.time()
    gradients4 = nn.backpropagation_with_reverse_AD(X_train, y_train)
    end = time.time() - start
    gradients4_total += end
#     print("gradients4", gradients4[0][0], gradients4[0][1], gradients4[1])

    nn.update_weights(gradients4, lr=0.01)
    epoch += 1


    # print("Epoch", epoch)

if epoch == max_epochs:
    print("Failed to converge")

backprop_time = time.time() - start_time
print(f"Training with backpropagation took {backprop_time:.2f} seconds")

print(f"Forward AD took {gradients3_total:.5f} seconds")
print(f"Reverse AD took {gradients4_total:.5f} seconds")

# Plot decision boundary
plot_decision_boundary(X_test, y_test, nn)
