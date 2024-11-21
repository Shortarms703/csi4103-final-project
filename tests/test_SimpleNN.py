import numpy as np

from FNN_multiple_layer_parameterized import SimpleNN, sigmoid


def test_layer_sizes_single_output_no_hidden_layer():
    nn = SimpleNN(2, [], 1)

    assert len(nn.W_list) == 1
    assert len(nn.b_list) == 1

    assert nn.W_list[0].shape == (2, 1)
    assert nn.b_list[0].shape == (1, 1)

    nn = SimpleNN(3, [], 1)

    assert len(nn.W_list) == 1
    assert len(nn.b_list) == 1

    assert nn.W_list[0].shape == (3, 1)
    assert nn.b_list[0].shape == (1, 1)

def test_layer_sizes_multi_output_no_hidden_layer():
    nn = SimpleNN(2, [], 2)

    assert len(nn.W_list) == 1
    assert len(nn.b_list) == 1

    assert nn.W_list[0].shape == (2, 2)
    assert nn.b_list[0].shape == (1, 2)

    nn = SimpleNN(3, [], 3)

    assert len(nn.W_list) == 1
    assert len(nn.b_list) == 1

    assert nn.W_list[0].shape == (3, 3)
    assert nn.b_list[0].shape == (1, 3)

def test_layer_sizes_single_output_single_hidden_layer():
    nn = SimpleNN(2, [2], 1)

    assert len(nn.W_list) == 2
    assert len(nn.b_list) == 2

    assert nn.W_list[0].shape == (2, 2)
    assert nn.b_list[0].shape == (1, 2)
    assert nn.W_list[1].shape == (2, 1)
    assert nn.b_list[1].shape == (1, 1)

    nn = SimpleNN(3, [3], 1)

    assert len(nn.W_list) == 2
    assert len(nn.b_list) == 2

    assert nn.W_list[0].shape == (3, 3)
    assert nn.b_list[0].shape == (1, 3)
    assert nn.W_list[1].shape == (3, 1)
    assert nn.b_list[1].shape == (1, 1)

def test_layer_sizes_multi_output_single_hidden_layer():
    nn = SimpleNN(2, [2], 2)

    assert nn.W_list[0].shape == (2, 2)
    assert nn.b_list[0].shape == (1, 2)
    assert nn.W_list[1].shape == (2, 2)
    assert nn.b_list[1].shape == (1, 2)

    nn = SimpleNN(3, [3], 3)

    assert nn.W_list[0].shape == (3, 3)
    assert nn.b_list[0].shape == (1, 3)
    assert nn.W_list[1].shape == (3, 3)
    assert nn.b_list[1].shape == (1, 3)

def test_layer_sizes_single_output_multi_hidden_layer():
    nn = SimpleNN(2, [2, 2], 1)

    assert len(nn.W_list) == 3
    assert len(nn.b_list) == 3

    assert nn.W_list[0].shape == (2, 2)
    assert nn.b_list[0].shape == (1, 2)
    assert nn.W_list[1].shape == (2, 2)
    assert nn.b_list[1].shape == (1, 2)
    assert nn.W_list[2].shape == (2, 1)
    assert nn.b_list[2].shape == (1, 1)

    nn = SimpleNN(3, [3, 2], 1)

    assert len(nn.W_list) == 3
    assert len(nn.b_list) == 3

    assert nn.W_list[0].shape == (3, 3)
    assert nn.b_list[0].shape == (1, 3)
    assert nn.W_list[1].shape == (3, 2)
    assert nn.b_list[1].shape == (1, 2)
    assert nn.W_list[2].shape == (2, 1)
    assert nn.b_list[2].shape == (1, 1)

def test_layer_sizes_multi_output_multi_hidden_layer():
    nn = SimpleNN(2, [2, 2], 2)

    assert len(nn.W_list) == 3
    assert len(nn.b_list) == 3

    assert nn.W_list[0].shape == (2, 2)
    assert nn.b_list[0].shape == (1, 2)
    assert nn.W_list[1].shape == (2, 2)
    assert nn.b_list[1].shape == (1, 2)
    assert nn.W_list[2].shape == (2, 2)
    assert nn.b_list[2].shape == (1, 2)

    nn = SimpleNN(3, [3, 2], 3)

    assert len(nn.W_list) == 3
    assert len(nn.b_list) == 3

    assert nn.W_list[0].shape == (3, 3)
    assert nn.b_list[0].shape == (1, 3)
    assert nn.W_list[1].shape == (3, 2)
    assert nn.b_list[1].shape == (1, 2)
    assert nn.W_list[2].shape == (2, 3)
    assert nn.b_list[2].shape == (1, 3)

def test_layer_sizes_many_layers():
    nn = SimpleNN(2, [5, 4, 5, 3, 2], 1)

    assert len(nn.W_list) == 6
    assert len(nn.b_list) == 6

    assert nn.W_list[0].shape == (2, 5)
    assert nn.b_list[0].shape == (1, 5)
    assert nn.W_list[1].shape == (5, 4)
    assert nn.b_list[1].shape == (1, 4)
    assert nn.W_list[2].shape == (4, 5)
    assert nn.b_list[2].shape == (1, 5)
    assert nn.W_list[3].shape == (5, 3)
    assert nn.b_list[3].shape == (1, 3)
    assert nn.W_list[4].shape == (3, 2)
    assert nn.b_list[4].shape == (1, 2)
    assert nn.W_list[5].shape == (2, 1)
    assert nn.b_list[5].shape == (1, 1)

def test_forward_no_hidden_layer():
    nn = SimpleNN(2, [], 1)

    nn.W_list[0] = np.array([[1], [2]])
    nn.b_list[0] = np.array([[3]])

    X = np.array([[4, 5], [6, 7]])

    y_pred = nn.forward(X)

    assert y_pred.shape == (2, 1)

    pre_sigmoid = nn.z_list  # pre sigmoid
    assert np.array_equal(pre_sigmoid[0], np.array([[1 * 4 + 2 * 5 + 3], [1 * 6 + 2 * 7 + 3]]))

    assert np.allclose(y_pred, np.array([[sigmoid(1 * 4 + 2 * 5 + 3)], [sigmoid(1 * 6 + 2 * 7 + 3)]]))

def test_forward_single_hidden_layer():
    nn = SimpleNN(2, [2], 1)

    nn.W_list[0] = np.array([[1, 2], [3, 4]])
    nn.b_list[0] = np.array([[5, 6]])

    nn.W_list[1] = np.array([[7], [8]])
    nn.b_list[1] = np.array([[9]])

    X = np.array([[10, 11], [12, 13]])

    y_pred = nn.forward(X)

    assert y_pred.shape == (2, 1)

    pre_sigmoid = nn.z_list

    pre_sigmoid_output_of_node_1_for_x1 = 1 * 10 + 3 * 11 + 5
    pre_sigmoid_output_of_node_2_for_x1 = 2 * 10 + 4 * 11 + 6

    assert np.array_equal(pre_sigmoid[0][0], np.array([pre_sigmoid_output_of_node_1_for_x1, pre_sigmoid_output_of_node_2_for_x1]))

    pre_sigmoid_output_of_node_1_for_x2 = 1 * 12 + 3 * 13 + 5
    pre_sigmoid_output_of_node_2_for_x2 = 2 * 12 + 4 * 13 + 6

    assert np.array_equal(pre_sigmoid[0][1], np.array([pre_sigmoid_output_of_node_1_for_x2, pre_sigmoid_output_of_node_2_for_x2]))

    pre_sigmoid_output_of_node_3_for_x1 = 7 * sigmoid(pre_sigmoid_output_of_node_1_for_x1) + 8 * sigmoid(pre_sigmoid_output_of_node_2_for_x1) + 9
    pre_sigmoid_output_of_node_3_for_x2 = 7 * sigmoid(pre_sigmoid_output_of_node_1_for_x2) + 8 * sigmoid(pre_sigmoid_output_of_node_2_for_x2) + 9

    assert np.array_equal(pre_sigmoid[1], np.array([[pre_sigmoid_output_of_node_3_for_x1], [pre_sigmoid_output_of_node_3_for_x2]]))

    assert np.allclose(y_pred, np.array([[sigmoid(pre_sigmoid_output_of_node_3_for_x1)], [sigmoid(pre_sigmoid_output_of_node_3_for_x2)]]))



