import numpy as np
from matplotlib import pyplot as plt


def test_accuracy(y_test, y_pred_test):
    # Apply threshold of 0.5 to convert probabilities to binary predictions
    y_pred_class = (y_pred_test > 0.5).astype(int)

    # Calculate accuracy
    accuracy = np.mean(y_pred_class == y_test)
    return accuracy


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title("Decision Boundary")
    plt.show()
