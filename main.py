import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_graph(graph, positions, output_path, labels=None):

    # node_shape_map = shape_split_vertices(graph)  # CAN ONLY BE IMPLEMEMENTED USING MULTIPLE PASSES

    # Draw Graph Embedding
    plt.figure(3, figsize=(20, 20))
    nx.draw(G=graph, pos=positions, node_shape='o', node_size=75)
    nx.draw_networkx_labels(G=graph, pos=positions, labels=labels, font_size=50)
    plt.savefig(fname=output_path, dpi=300)
    plt.clf()


def planarize_graph(graph, positions, edge_crossings, largest_index=None):

    # Extract basic properties of graph
    # TODO: THIS INDEX BREAK EVERYTHING SOMEHOW
    index = largest_index if largest_index is not None else max(graph.nodes)
    edges = list(graph.edges)  # create list for easier indexing

    edge_to_virtual_vertex = {edge: set() for edge in edges}  # have to ensure
    edges_to_be_removed = set()  # could be initialized using size of dictionary 'edge_crossings'

    # Iterate over all found edge crossings
    for edge_a in edge_crossings.keys():
        for edge_b in edge_crossings[edge_a].keys():

            # Update index
            index += 1

            # Add new vertex to graph and drawing's locations
            graph.add_node(node_for_adding=index, virtual=1)
            positions[index] = np.asarray(edge_crossings[edge_a][edge_b])

            # Log connections to new virtual vertex to be added and original (real) edges to be removed
            [edge_to_virtual_vertex[edge].add(index) for edge in [edge_a, edge_b]]
            [edges_to_be_removed.add(edge) for edge in [edge_a, edge_b]]

    # Remove original edge set and add virtual edge set
    virtual_edge_set = add_virtual_edges(graph, positions, edge_to_virtual_vertex)
    remove_edges(graph, list(edges_to_be_removed))

    #  return some new graph and new vertex positions
    return virtual_edge_set


def project_point_onto_line(point, start_point, end_point):
    normalized = (end_point-start_point) / np.linalg.norm(x=end_point-start_point)
    return np.dot(point - start_point, normalized)


def sort_vertices_along_edge(edge, vertex_set, positions):

    # Extract the 2D coordinates of the edge line
    start_vertex, end_vertex = edge[0], edge[1]
    start_point, end_point = positions[start_vertex], positions[end_vertex]

    # Initialize container for magnitudes from edge start-point
    projections = np.empty(len(vertex_set))

    # Iterate over all vertices, including start and end points of the edge
    # vertices_to_sort = list(vertex_set) + [start_vertex, end_vertex]
    for index, vertex in enumerate(vertex_set):
        projections[index] = project_point_onto_line(positions[vertex], start_point, end_point)

    # Sort indices of projections and sort vertex indices
    sorted_indices = np.argsort(projections)
    sorted_vertices = [vertex_set[i] for i in sorted_indices]

    # Ensure Start and End-Points sorted correctly (i.e first and last)
    sorted_vertices = [start_vertex] + sorted_vertices + [end_vertex]

    # Return sorted vertex indices
    return sorted_vertices

def add_virtual_edges(graph, positions, edge_to_virtual_vertex):

    # A Map of virtual edges which describe the same edge
    virtual_edge_set = {edge: [] for edge in edge_to_virtual_vertex}

    # Iterate over all edges in the graph
    for edge in edge_to_virtual_vertex.keys():

        edge_data = graph.get_edge_data(v=edge[0], u=edge[1], default={})
        # Skip edge if it does not have any edge crossings
        if len(edge_to_virtual_vertex[edge]) == 0:
            continue

        # Extract all the virtual vertices and (together with real edge points) sort them
        virtual_vertices = list(edge_to_virtual_vertex[edge])
        sorted_vertex_targets = sort_vertices_along_edge(edge, virtual_vertices, positions)

        # Connect vertices in sort order (undirected edges so order doesn't matter)
        for index in range(1, len(sorted_vertex_targets)):
            vertex_a, vertex_b = sorted_vertex_targets[index-1], sorted_vertex_targets[index]
            virtual_edge_set[edge].append((vertex_a, vertex_b))
            graph.add_edge(u_of_edge=vertex_a,
                           v_of_edge=vertex_b,
                           virtual=1)
    return virtual_edge_set

def remove_edges(graph, edges_to_be_removed):
    for edge in edges_to_be_removed:
        graph.remove_edge(u=edge[0], v=edge[1])


def locate_edge_crossings(graph, positions):

    # Create object of edges for easier use
    edges = list(graph.edges)

    # Initialize vector and edge crossing containers
    vertex_crossings = {vertex: 0 for vertex in graph.nodes()}
    edge_crossings = dict()

    # ALl to all comparison of edges
    for edge_index_a in range(0, len(edges)):
        for edge_index_b in range(edge_index_a + 1, len(edges)):

            # Extract edges from edge list
            edge_a = edges[edge_index_a]
            edge_b = edges[edge_index_b]

            # Check if the two edges share a common vertex (causes numerical issues)
            if (edge_a[0] in edge_b) or (edge_a[1] in edge_b):
                continue

            # Check whether edges intersect and (if so) where
            intersection = edge_intersection(edge_a, edge_b, positions)
            if intersection is None:
                continue

            # Append edge crossing position for edges
            if edge_a not in edge_crossings:
                edge_crossings[edge_a] = dict()
            edge_crossings[edge_a][edge_b] = intersection

            # Increment edge crossing count for all vertices involves in crossing
            crossing_vertices = np.append(np.asarray(edge_a), np.asarray(edge_b))
            for vertex in crossing_vertices:
                vertex_crossings[vertex] += 1

    #  return two dicts, one for vertices and one for edge
    return edge_crossings, vertex_crossings


def edge_intersection(edge_a, edge_b, vertex_positions):
    point_a_0 = vertex_positions[edge_a[0]]
    point_a_1 = vertex_positions[edge_a[1]]
    point_b_0 = vertex_positions[edge_b[0]]
    point_b_1 = vertex_positions[edge_b[1]]
    return line_intersection(point_a_0, point_a_1, point_b_0, point_b_1)


def line_intersection(p1, p2, p3, p4):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])

    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denominator == 0:  # parallel
        return None
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator

    # TODO: investigate these statements. just adding >= instead of > strikes me as dangerous
    if ua <= 0 or ua >= 1:
        return None
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

    # TODO: investigate these statements. just adding >= instead of > strikes me as dangerous
    if ub <= 0 or ub >= 1:
        return None
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return x, y


def connect_singleton_vertex_edges(graph, positions):

    for vertex in graph.nodes:
        if graph.degree(vertex) != 1:
            continue
        closest_target_vertex = find_closest_vertex(vertex, graph, positions)
        graph.add_edge(u_of_edge=vertex,
                       v_of_edge=closest_target_vertex,
                       virtual=1)
        print(f"connected {vertex} to {closest_target_vertex}")
        return connect_singleton_vertex_edges(graph, positions)


def squared_distance(point_a, point_b):
    (x1, y1) = point_a
    (x2, y2) = point_b
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def find_closest_edge_intersection(edge_points, other_edges, graph, positions, must_be_real=False):
    intersections, distances = dict(), dict()
    point_a, point_b = edge_points
    # print(f"Points a and b: {point_b} - {point_b}")
    for edge in other_edges:

        # Check whether the intersection is with a real edge
        if must_be_real:
            fields = graph.get_edge_data(v=edge[0], u=edge[1], default=None)
            if fields:
                if fields.get("real", 0) == 0:
                    continue
            else:
                print(f"fields of edge {edge[0], edge[1]}: {fields}")  # todo: certain edges don't have data, which shouldn't happen

        # Find
        # print(f"current edge: {edge[0], edge[1]}")
        point_c, point_d = positions[edge[0]], positions[edge[1]]
        # print(f"Points c and d: {point_c} - {point_d}")
        intersection = line_intersection(point_a, point_b, point_c, point_d)
        if intersection is None:
            continue
        intersections[edge] = intersection
        distances[edge] = squared_distance(point_a, intersection)

    # If no intersection was found, return none
    if len(intersections) == 0 and len(distances) == 0:
        return None, None

    # Find closest intersection
    closest_intersection = min(distances, key=distances.get)

    # Return the Edge Name, and it's intersection as a tuple
    return closest_intersection, intersections[closest_intersection]


def find_closest_vertex(vertex, graph, positions):
    edge_list = [edge for edge in graph.edges()]
    edge_sets = [frozenset(edge) for edge in edge_list]
    distances = {node: float("inf") for node in graph.nodes}
    for node in graph.nodes:
        if node == vertex:
            continue
        if {node, vertex} in edge_sets:
            continue

        hypothetical_edge = (positions[vertex], positions[node])
        closest_intersection, intersections = find_closest_edge_intersection(edge_points=hypothetical_edge,
                                                                             other_edges=edge_list,
                                                                             graph=graph,
                                                                             positions=positions)
        if closest_intersection is not None:
            continue

        distances[node] = squared_distance(point_a=positions[vertex],
                                           point_b=positions[node])
    print(distances)
    closest_vertex = min(distances, key=distances.get)
    return closest_vertex


def find_inner_faces(remaining_cycles, identified_faces, sub_graph, sub_positions):


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

    #
    edge_crossings, vertex_crossings = locate_edge_crossings(graph=graph, positions=positions)
    virtual_edge_set = planarize_graph(graph=graph, positions=positions, edge_crossings=edge_crossings)

    draw_graph(graph=graph, positions=positions, output_path="./planar_graph.png")

    connect_singleton_vertex_edges(graph=graph, positions=positions)
    draw_graph(graph=graph, positions=positions, output_path="./closed_graph.png")

    cycles = nx.minimum_cycle_basis(G=graph)
    print(cycles)
