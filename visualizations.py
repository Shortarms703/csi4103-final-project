import networkx as nx
import matplotlib.pyplot as plt


def visualize_nn(layer_sizes):
    G = nx.DiGraph()

    # Create nodes
    node_sizes = {}
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            node_id = f'L{i}_N{j}'
            G.add_node(node_id, layer=i, pos=(i, -j))
            node_sizes[node_id] = 500

    # Create edges
    for i, (size_a, size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for j in range(size_a):
            for k in range(size_b):
                G.add_edge(f'L{i}_N{j}', f'L{i + 1}_N{k}')

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold',
            edge_color='gray')
    plt.show()


# Define network structure
layer_sizes = [2, 3, 1]
visualize_nn(layer_sizes)

