import itertools
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
    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        """
        Initialize weights
        :param input_size:
        :param output_size:
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        all_layers = [input_size, *hidden_layers, output_size]

        self.W_list = []
        self.b_list = []

        for i in range(len(all_layers) - 1):
            self.W_list.append(np.random.randn(all_layers[i], all_layers[i + 1]))
            self.b_list.append(np.zeros((1, all_layers[i + 1])))


        self.z_list = []  # before activation function
        self.a_list = []  # after activation function

        self.timer = 0

    def reset_timer(self):
        self.timer = 0

    def get_timer(self):
        return self.timer

    def forward(self, X):
        """
        Forward pass
        :param X:
        :return:
        """

        self.z_list = []
        self.a_list = [X]

        for W, b in zip(self.W_list, self.b_list):
            z = np.dot(self.a_list[-1], W) + b
            self.z_list.append(z)
            a = sigmoid(z)
            self.a_list.append(a)

        return self.a_list[-1]


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

    # def backpropagation_gradients(self, X, y):
    #     """
    #     Calculates gradients using backpropagation (typical approach)
    #     :param X:
    #     :param y:
    #     :return:
    #     """
    #     m = X.shape[0]
    #
    #     y_pred = self.forward(X)
    #
    #     dz1 = y_pred - y  # dL/dz1
    #
    #     dW1 = np.dot(X.T, dz1) / m  # dL/dW1
    #     db1 = np.sum(dz1, axis=0, keepdims=True) / m  # dL/db1
    #
    #     return dW1, db1

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


    def get_inputs_weights_biases(self):
        inputs = [Variable(f'X_{i}') for i in range(1, self.input_size + 1)]
        outputs = [Variable(f'y_{i}') for i in range(1, self.output_size + 1)]

        all_layers = [self.input_size, *self.hidden_layers, self.output_size]
        weights = []
        biases = []
        for i in range(len(all_layers) - 1):
            weights.append([Variable(f'W_{i}_{j}') for j in range(all_layers[i])])
            biases.append([Variable(f'b_{i}_{j}') for j in range(all_layers[i])])

        return inputs, outputs, list(itertools.chain(*weights)), list(itertools.chain(*biases))


    def get_output_equation(self):
        # all assumes 1 output node
        inputs, outputs, weights_1, biases_1 = self.get_inputs_weights_biases()

        to_sum = [Multiply(inputs[i], weights_1[i]) for i in range(self.input_size)]
        s_z1 = None
        for i in range(1, self.input_size):
            if s_z1 is None:
                s_z1 = Add(to_sum[i - 1], to_sum[i])
            else:
                s_z1 = Add(s_z1, to_sum[i])
        s_z1 = Add(s_z1, biases_1[0])

        s_sigmoid_loss = Expression(Exponent(Add(Sigmoid(s_z1), Multiply(Constant(-1), outputs[0])), Constant(2)))

        return s_sigmoid_loss

    def create_backpropagation_values(self, X, y, inputs, outputs, weights_1, biases_1, partials):
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)

        for i in range(X.shape[0]):
            y_val = y[i, 0]
            values_dict = {inputs[j].name: X[i][j] for j in range(self.input_size)}
            values_dict.update({weights_1[j].name: self.W1[j, 0] for j in range(self.input_size)})
            values_dict.update({biases_1[0].name: self.b1[0, 0]})
            values_dict.update({outputs[0].name: y_val})

            start = time.time()
            for j in range(self.input_size):
                dW1[j, 0] += partials[j].evaluate(values_dict)

            db1[0, 0] += partials[self.input_size].evaluate(values_dict)
            self.timer += time.time() - start

        dW1 /= X.shape[0]
        db1 /= X.shape[0]

        return dW1, db1

    def backpropagation_with_forward_AD(self, X, y):
        """
        Backpropagation with gradients calculated with forward accumulation
        :param X:
        :param y:
        :return:
        """
        inputs, outputs, weights_1, biases_1 = self.get_inputs_weights_biases()
        s_sigmoid_loss = self.get_output_equation()

        partials = [forward_propagation_partial(s_sigmoid_loss, weights_1[i]) for i in range(self.input_size)]
        partials += [forward_propagation_partial(s_sigmoid_loss, biases_1[0])]

        return self.create_backpropagation_values(X, y, inputs, outputs, weights_1, biases_1, partials)

    def backpropagation_with_reverse_AD(self, X, y):
        """
        Backpropagation with gradients calculated with reverse accumulation
        :param X:
        :param y:
        :return:
        """
        inputs, outputs, weights_1, biases_1 = self.get_inputs_weights_biases()
        s_sigmoid_loss = self.get_output_equation()

        partials_object = back_propagation_partial(s_sigmoid_loss)
        partials = [partials_object.get_derivative(i) for i in [*weights_1, *biases_1]]

        return self.create_backpropagation_values(X, y, inputs, outputs, weights_1, biases_1, partials)


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


def generate_linearly_separable_data(n_samples_per_class, n_dimensions, separation=2.0, noise=1.0):
    """
    Generates linearly separable data in n-dimensional space.
    :param n_samples_per_class:
    :param n_dimensions:
    :param separation:
    :param noise:
    :return:
    """
    # Generate a random hyperplane normal vector
    hyperplane_normal = np.random.randn(n_dimensions)
    hyperplane_normal /= np.linalg.norm(hyperplane_normal)

    # Generate points for class 1 (positive class)
    X1 = np.random.randn(n_samples_per_class, n_dimensions) * noise
    X1 += separation * hyperplane_normal  # Shift points above the hyperplane

    # Generate points for class 2 (negative class)
    X2 = np.random.randn(n_samples_per_class, n_dimensions) * noise
    X2 -= separation * hyperplane_normal  # Shift points below the hyperplane

    # Combine the data
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n_samples_per_class), np.zeros(n_samples_per_class)])

    return X, y

if __name__ == '__main__':
    # np.random.seed(3)

    dimensions = 5
    X, y = generate_linearly_separable_data(500, dimensions, separation=1, noise=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    nn = SimpleNN(dimensions, 1)

    gradients3_total = 0
    gradients4_total = 0

    start_time = time.time()
    accuracy = 0
    epoch = 0
    max_epochs = 10000
    plot = (dimensions == 2)

    # while accuracy < 0.99 and epoch < max_epochs:
    while epoch < 10000:
        if epoch % 100 == 0:
            y_pred_test = nn.forward(X_test)
            print(f"Epoch {epoch}")
            accuracy = test_accuracy(y_test, y_pred_test)
            print(f"Test accuracy: {accuracy:.2f}")
            loss = SimpleNN.loss(y_test, y_pred_test)
            print(f"Test loss: {loss:.2f}")
            if plot:
                plot_decision_boundary(X_test, y_test, nn)
                time.sleep(0.5)
            print(f"Forward AD taken {gradients3_total:.5f} seconds so far")
            print(f"Reverse AD taken {gradients4_total:.5f} seconds so far")
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

    if epoch == max_epochs and accuracy < 0.99:
        print("Failed to converge")

    backprop_time = time.time() - start_time
    print(f"Training with backpropagation took {backprop_time:.2f} seconds")

    print(f"Forward AD took {gradients3_total:.5f} seconds")
    print(f"Reverse AD took {gradients4_total:.5f} seconds")

    # Plot decision boundary
    if plot:
        plot_decision_boundary(X_test, y_test, nn)
