import time

from sklearn.model_selection import train_test_split

from FNN_single_layer_parameterized import SimpleNN, generate_linearly_separable_data
from utils import test_accuracy, plot_decision_boundary

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

backprop_time = time.time() - start_time
print(f"Training with backpropagation took {backprop_time:.2f} seconds")

print(f"Forward AD took {gradients3_total:.5f} seconds")
print(f"Reverse AD took {gradients4_total:.5f} seconds")
