import sys
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy

def draw_graph(graph, positions, output_path, labels=None):

    # node_shape_map = shape_split_vertices(graph)  # CAN ONLY BE IMPLEMEMENTED USING MULTIPLE PASSES

    # Draw Graph Embedding
    plt.figure(3, figsize=(30, 30))
    nx.draw(G=graph, pos=positions, node_shape='o', node_size=25)
    nx.draw_networkx_labels(G=graph, pos=positions, labels=labels, font_size=15)
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
        print(f"Virtually connected {vertex} to {closest_target_vertex}")
        return connect_singleton_vertex_edges(graph, positions)


def squared_distance(point_a, point_b):
    (x1, y1) = point_a
    (x2, y2) = point_b
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def any_intersections(main_edge, edge_list, positions):

    point_a, point_b = positions[main_edge[0]], positions[main_edge[1]]

    # print(f"Points a and b: {point_b} - {point_b}")
    for edge in edge_list:

        if any([vertex in main_edge for vertex in edge]):
            continue

        # Find
        point_c, point_d = positions[edge[0]], positions[edge[1]]
        intersection = line_intersection(point_a, point_b, point_c, point_d)
        if intersection is not None:
            return True

    return False


def find_closest_vertex(vertex, graph, positions):
    edge_list = [edge for edge in graph.edges()]
    edge_sets = [frozenset(edge) for edge in edge_list]
    distances = {node: float("inf") for node in graph.nodes}
    for node in graph.nodes:

        if node == vertex:
            continue
        if {node, vertex} in edge_sets:
            continue

        if any_intersections(main_edge=(vertex, node), edge_list=edge_list, positions=positions):
            continue

        distances[node] = squared_distance(point_a=positions[vertex], point_b=positions[node])

    closest_vertex = min(distances, key=distances.get)
    return closest_vertex


def get_cycle_edges(cycle, graph):

    # Extract Subgraph of only the vertices of the cycle
    sub_graph = graph.copy()
    sub_graph.remove_nodes_from([node for node in graph.nodes if node not in cycle])

    # Count the degree of all vertices in the subgraph.
    degrees = {node: sub_graph.degree(node) for node in sub_graph.nodes}
    problem_nodes = set([node for node, degree in degrees.items() if degree > 2])

    #
    assert (len(problem_nodes) == 0), \
        f"Not all vertices have degree of two: {problem_nodes}"

    # Return Edges as list of frozensets
    return [frozenset(edge) for edge in sub_graph.edges]


def order_cycle_vertices(cycle, graph):
    cycle_edges = get_cycle_edges(cycle=cycle, graph=graph)
    ordered_cycle_edges = get_ordered_edges(edges=cycle_edges)
    vertex_sequence = get_vertex_sequence(edges=ordered_cycle_edges, is_ordered=True)
    return vertex_sequence


def get_ordered_edges(edges):

    edges = [tuple(edge) for edge in edges]

    #
    sorted_edges = [(None, None)] * len(edges)
    sorted_edges[0] = edges[0]
    visited_edges = [{edges[0]}]

    #
    for i in range(1, len(edges)):
        for edge in edges:
            if {edge} in visited_edges:
                continue
            if edge[0] == sorted_edges[i-1][1] and (edge[0], edge[1] not in sorted_edges):
                sorted_edges[i] = edge
                visited_edges.append({edge})
            elif edge[1] == sorted_edges[i-1][1] and (edge[1], edge[0] not in sorted_edges):
                sorted_edges[i] = (edge[1], edge[0])
                visited_edges.append({edge})

    #
    return sorted_edges


def place_virtual_midpoints(graph, positions, start_index=None):
    start_index = start_index if start_index is not None else max(graph.nodes()) + 1
    for index, (node_a, node_b) in enumerate(copy.deepcopy(graph.edges)):
        new_vertex = start_index + index
        positions[new_vertex] = calculate_midpoint(positions[node_a], positions[node_b])
        graph.add_edge(u_of_edge=node_a, v_of_edge=new_vertex)
        graph.add_edge(u_of_edge=node_b, v_of_edge=new_vertex)
        graph.remove_edge(v=node_a, u=node_b)


def calculate_midpoint(point_a, point_b):
    return (point_a[0] + point_b[0])/2.0, (point_a[1] + point_b[1])/2.0


def get_vertex_sequence(edges, is_ordered=False):
    edges = get_ordered_edges(edges=edges) if not is_ordered else edges
    vertex_sequence = [edge[0] for edge in edges]
    return vertex_sequence


def is_cycle_empty(ordered_cycle, graph, positions):

    #
    ordered_cycle_closed = ordered_cycle + [ordered_cycle[0]]
    ordered_coordinates = [positions[cycle_node] for cycle_node in ordered_cycle_closed]

    #
    cycle_path = mpltPath.Path(vertices=ordered_coordinates, codes=None, closed=True, readonly=True)

    remaining_nodes = [node for node in graph.nodes if node not in ordered_cycle]
    in_side = cycle_path.contains_points([positions[node] for node in remaining_nodes])

    #
    return any(in_side)


def count_cycle_per_edge(cycles, graph):
    edge_counts = {frozenset(edge): 0 for edge in graph.edges}
    for cycle in cycles:
        cycle_edges = get_cycle_edges(cycle=cycle, graph=graph)
        for edge in cycle_edges:
            edge_counts[edge] += 1
    return edge_counts


def get_faces(cycles, original_vertices, graph):
    ordered_faces = []
    for cycle in cycles:
        ordered_cycle_vertices = order_cycle_vertices(cycle=cycle, graph=graph)
        ordered_faces.append([vertex for vertex in ordered_cycle_vertices if vertex in original_vertices])
    return ordered_faces


def count_cycles_edge(cycle_edges, graph):
    graph_edge_counts = {frozenset(edge): 0 for edge in graph.edges}
    for cycle, cycle_edges in cycle_edges.items():
        for cycle_edge in cycle_edges:
            cycle_edge = frozenset(cycle_edge)
            if cycle_edge in graph_edge_counts.keys():
                graph_edge_counts[cycle_edge] += 1
    return graph_edge_counts


def prune_cycle_graph(cycle_edges, graph):
    edge_counts = count_cycles_edge(cycle_edges, graph)
    # TODO: since we are checking against ALL cycles, it is indeed possible for one edge to map MORE than twice
    #  these are incorrect cycles (i.e. not faces), hence the messy mapping
    mapped_edges = [tuple(edge) for edge, count in edge_counts.items() if count == 2]
    [print(f"{edge} - {count}") for edge, count in edge_counts.items()]
    input("COUNTS ARE FUCKED")
    graph.remove_edges_from(mapped_edges)


def get_nodes_in_cycle(ordered_cycle, graph, positions):

    #
    ordered_cycle_closed = ordered_cycle + [ordered_cycle[0]]
    ordered_coordinates = [positions[cycle_node] for cycle_node in ordered_cycle_closed]

    #
    cycle_path = mpltPath.Path(vertices=ordered_coordinates, codes=None, closed=True, readonly=True)

    #
    remaining_nodes = [node for node in graph.nodes if node not in ordered_cycle]
    in_side = cycle_path.contains_points([positions[node] for node in remaining_nodes])

    #
    return [node for index, node in enumerate(remaining_nodes) if in_side[index]]


def identify_faces(faces, graph, positions):

    # Identify the minimum cycle basis of the graph
    cycles = [frozenset(cycle) for cycle in nx.minimum_cycle_basis(G=graph)]
    ordered_edges = {cycle: get_ordered_edges(get_cycle_edges(cycle=cycle, graph=graph)) for cycle in cycles}
    ordered_nodes = {cycle: get_vertex_sequence(edges=ordered_edges[cycle], is_ordered=True) for cycle in cycles}

    #
    for cycle in cycles:
        if cycle in faces:
            continue

        #
        nodes_inside_cycle = get_nodes_in_cycle(ordered_nodes[cycle], graph, positions)
        if len(nodes_inside_cycle) == 0:
            faces.add(cycle)
        else:
            nodes_inside_cycle += list(cycle)
            sub_graph = graph.subgraph(nodes=list(set(nodes_inside_cycle))).copy()
            prune_cycle_graph(cycle_edges=ordered_edges, graph=sub_graph)
            sub_positions = {node: ordered_nodes.get(node) for node in nodes_inside_cycle}
            identify_faces(faces, sub_graph, sub_positions)


def get_sorted_face_vertices_from_cycle(ordered_cycle_nodes, original_vertices):
    ordered_faces = []
    for cycle, ordered_cycle_vertices in ordered_cycle_nodes.items():
        ordered_faces.append([vertex for vertex in ordered_cycle_vertices if vertex in original_vertices])
    return ordered_faces


def get_ordered_edges_from_ordered_vertices(ordered_vertices: []):
    edges = [(None, None)] * len(ordered_vertices)
    for index in range(0, len(ordered_vertices)):
        edges[index] = (ordered_vertices[index], ordered_vertices[(index + 1) % len(ordered_vertices)])
    return edges


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #
    cmd_args = sys.argv
    n_vertices, m_edges, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    # Simulate Graph
    original_graph = nx.barabasi_albert_graph(n=n_vertices, m=m_edges, seed=seed)
    original_positions = nx.kamada_kawai_layout(G=original_graph)
    draw_graph(graph=original_graph, positions=original_positions, output_path="./original_graph.png")

    # Planarize Graph by removing edge crossings and replacing them with virtual vertices
    planar_graph, planar_positions = copy.deepcopy(original_graph), copy.deepcopy(original_positions)
    edge_crossings, vertex_crossings = locate_edge_crossings(graph=original_graph, positions=original_positions)
    virtual_edge_set = planarize_graph(graph=planar_graph, positions=planar_positions, edge_crossings=edge_crossings)
    face_vertices = list(planar_graph.nodes)
    draw_graph(graph=planar_graph, positions=planar_positions, output_path="./planar_graph.png")

    # Add virtual edge to bridge-vertices to ensure all vertices can be a part of a cycle
    cyclical_graph, cyclical_positions = copy.deepcopy(planar_graph), copy.deepcopy(planar_positions)
    connect_singleton_vertex_edges(graph=cyclical_graph, positions=cyclical_positions)
    labels = {vertex: vertex for vertex in cyclical_graph.nodes}
    draw_graph(graph=cyclical_graph, positions=cyclical_positions, output_path="./closed_graph.png")

    # Create midpoint graph by replacing all edges with two virtual edges with a vertex at its center
    midpoint_graph, midpoint_positions = copy.deepcopy(cyclical_graph), copy.deepcopy(cyclical_positions)
    place_virtual_midpoints(graph=midpoint_graph, positions=midpoint_positions)
    draw_graph(graph=midpoint_graph, positions=midpoint_positions, output_path="./expanded_graph.png")

    # Identify Faces from Cycles
    identified_faces = set()
    identify_faces(faces=identified_faces, graph=midpoint_graph, positions=midpoint_positions)

    # Check Face Edge Counts
    outer_graph, outer_positions = copy.deepcopy(midpoint_graph), copy.deepcopy(midpoint_positions)
    edge_counts = count_cycle_per_edge(cycles=identified_faces, graph=outer_graph)
    for edge, count in edge_counts.items():
        if count == 2:
            edge = set(edge)
            outer_graph.remove_edge(u=edge.pop(), v=edge.pop())
    draw_graph(graph=outer_graph, positions=outer_positions, output_path="./outer_graph.png")

    # Identify Each Face's Ordered Edge and Node List from the mid-point graph
    unsorted_edges = {face: get_cycle_edges(cycle=face, graph=midpoint_graph) for face in identified_faces}
    ordered_edges = {face: get_ordered_edges(unsorted_edges.get(face)) for face in identified_faces}
    ordered_nodes = {face: get_vertex_sequence(edges=ordered_edges[face], is_ordered=True) for face in identified_faces}

    # Clean up Sorted Vertex and Edge lists from midpoint vertices
    sorted_faces = get_sorted_face_vertices_from_cycle(ordered_cycle_nodes=ordered_nodes,
                                                       original_vertices=face_vertices)

    # Extract the faces, and their sorted edges and vertices
    faces = set([frozenset(sorted_face) for sorted_face in sorted_faces])
    sorted_face_vertices = {frozenset(face): face for face in sorted_faces}
    sorted_face_edges = {frozenset(face): get_ordered_edges_from_ordered_vertices(face) for face in sorted_faces}

    # Return Identified Faces and their sorted Edges
    [print(f"{face} - {edges}") for face, edges in sorted_face_edges.items()]


