import sys
import networkx as nx
import numpy as np


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cmd_args = sys.argv
    n_vertices, m_edges, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    graph = nx.barabasi_albert_graph(n=n_vertices, m=m_edges, seed=seed)
    positions = nx.kamada_kawai_layout(G=graph)

