import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_graph(graph, positions, output_path, labels=None):

    # node_shape_map = shape_split_vertices(graph)  # CAN ONLY BE IMPLEMEMENTED USING MULTIPLE PASSES

    # Draw Graph Embedding
    plt.figure(3, figsize=(20, 20))
    nx.draw(G=graph, pos=positions, node_shape='o', node_size=75)
    if labels is not None:
        nx.draw_networkx_labels(G=graph, pos=positions, labels=labels, font_size=15)
    else:
        nx.draw_networkx_labels(G=graph, pos=positions, font_size=15)

    plt.savefig(fname=output_path, dpi=300)
    plt.clf()


def find_inner_faces(graph, positions):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = sys.argv
    n_vertices, m_edges, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # Simulate Graph
    graph = nx.barabasi_albert_graph(n=n_vertices, m=m_edges, seed=seed)

    # Embed Graph  in 2D
    positions = nx.kamada_kawai_layout(G=graph)

    #
    draw_graph(graph=graph, positions=positions, output_path="./graph.png")

    cycles = nx.minimum_cycle_basis(G=graph)
    print(cycles)
