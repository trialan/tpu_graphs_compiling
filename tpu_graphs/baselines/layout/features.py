from tqdm import tqdm
import tensorflow as tf
import networkx as nx
import numpy as np


def compute_closeness_centrality(edge_ranges, node_ranges, edge_index):
    """Compute Closeness Centrality for each node."""
    closeness_centrality_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="ClosenessCentrality"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        centrality = nx.closeness_centrality(G)
        centrality_values = [centrality.get(node, 0) for node in range(num_nodes)]
        closeness_centrality_features.extend(centrality_values)

    closeness_centrality_tensor = tf.convert_to_tensor(closeness_centrality_features, dtype=tf.float32)
    return tf.reshape(closeness_centrality_tensor, [-1, 1])



def compute_average_neighbor_degree(edge_ranges, node_ranges, edge_index):
    """Compute the average neighbor degree for each node."""
    avg_neighbor_degree_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="AvgNeighborDeg"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        avg_neighbor_degree = nx.average_neighbor_degree(G)

        # Ensure average neighbor degree values for all nodes, including isolated ones
        avg_neighbor_degree_values = [avg_neighbor_degree.get(node, 0) for node in range(num_nodes)]
        avg_neighbor_degree_features.extend(avg_neighbor_degree_values)

    avg_neighbor_degree_tensor = tf.convert_to_tensor(avg_neighbor_degree_features, dtype=tf.float32)
    return tf.reshape(avg_neighbor_degree_tensor, [-1, 1])


def compute_hits(edge_ranges, node_ranges, edge_index):
    """Compute HITS (Hubs and Authorities) scores for each node in a directed graph."""
    hubs_features = []
    authorities_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="HITS"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.DiGraph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        hubs, authorities = nx.hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True)

        # Ensure hubs and authorities values for all nodes, including isolated ones
        hubs_values = [hubs.get(node, 0) for node in range(num_nodes)]
        authorities_values = [authorities.get(node, 0) for node in range(num_nodes)]

        hubs_features.extend(hubs_values)
        authorities_features.extend(authorities_values)

    hubs_tensor = tf.convert_to_tensor(hubs_features, dtype=tf.float32)
    authorities_tensor = tf.convert_to_tensor(authorities_features, dtype=tf.float32)

    return tf.reshape(hubs_tensor, [-1, 1]), tf.reshape(authorities_tensor, [-1, 1])


def compute_out_degree_centrality(edge_ranges, node_ranges, edge_index):
    """Compute Out-Degree Centrality for each node in a directed graph."""
    out_degree_centrality_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="OutDegreeCentrality"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.DiGraph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        centrality = nx.out_degree_centrality(G)
        centrality_values = [centrality.get(node, 0) for node in range(num_nodes)]
        out_degree_centrality_features.extend(centrality_values)

    out_degree_centrality_tensor = tf.convert_to_tensor(out_degree_centrality_features, dtype=tf.float32)
    return tf.reshape(out_degree_centrality_tensor, [-1, 1])

def compute_in_degree_centrality(edge_ranges, node_ranges, edge_index):
    """Compute In-Degree Centrality for each node in a directed graph."""
    in_degree_centrality_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="InDegreeCentrality"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.DiGraph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        centrality = nx.in_degree_centrality(G)
        centrality_values = [centrality.get(node, 0) for node in range(num_nodes)]
        in_degree_centrality_features.extend(centrality_values)

    in_degree_centrality_tensor = tf.convert_to_tensor(in_degree_centrality_features, dtype=tf.float32)
    return tf.reshape(in_degree_centrality_tensor, [-1, 1])


def compute_degree_centrality(edge_ranges, node_ranges, edge_index):
    """Compute Degree Centrality for each node."""
    degree_centrality_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="Degree Centrality"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        centrality = nx.degree_centrality(G)
        # Ensure centrality values for all nodes, including isolated ones
        centrality_values = [centrality.get(node, 0) for node in range(num_nodes)]
        degree_centrality_features.extend(centrality_values)

    degree_centrality_tensor = tf.convert_to_tensor(degree_centrality_features, dtype=tf.float32)
    return tf.reshape(degree_centrality_tensor, [-1, 1])


def compute_structural_holes_metrics(edge_ranges, node_ranges, edge_index):
    """Compute structural holes metrics for each node."""
    constraint_features = []
    effective_size_features = []
    efficiency_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="Structural Holes"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)
        num_nodes = node_ranges[index + 1] - node_ranges[index]

        for node in range(num_nodes):
            if node in G:
                neighbors = set(G.neighbors(node))
                size = len(neighbors)
                local_constraint = 0

                for neighbor in neighbors:
                    proportion = 1 / size  # Assuming unweighted edges
                    shared_neighbors = set(G.neighbors(neighbor)) & neighbors
                    local_constraint += proportion * (1 + len(shared_neighbors))

                constraint = local_constraint
                effective_size = size - local_constraint
                efficiency = effective_size / size if size > 0 else 0
            else:
                # Default values for isolated nodes
                constraint = 0
                effective_size = 0
                efficiency = 0

            constraint_features.append(constraint)
            effective_size_features.append(effective_size)
            efficiency_features.append(efficiency)

    # Convert lists to TensorFlow tensors
    constraint_tensor = tf.convert_to_tensor(constraint_features, dtype=tf.float32)
    effective_size_tensor = tf.convert_to_tensor(effective_size_features, dtype=tf.float32)
    efficiency_tensor = tf.convert_to_tensor(efficiency_features, dtype=tf.float32)

    return tf.reshape(constraint_tensor, [-1, 1]), tf.reshape(effective_size_tensor, [-1, 1]), tf.reshape(efficiency_tensor, [-1, 1])


def compute_generalized_degree(edge_ranges, node_ranges, edge_index, distance=1):
    """Compute the generalized degree for each node."""
    generalized_degree_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="Generalized Degree"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)

        num_nodes = node_ranges[index + 1] - node_ranges[index]
        generalized_degree = dict()

        for node in range(num_nodes):
            if node in G:
                generalized_degree[node] = sum(1 for _ in nx.ego_graph(G, node, radius=distance, center=False))
            else:
                generalized_degree[node] = 0

        generalized_degree_values = [generalized_degree.get(node, 0) for node in range(num_nodes)]
        generalized_degree_features.extend(generalized_degree_values)

    generalized_degree_features = tf.convert_to_tensor(generalized_degree_features, dtype=tf.float32)
    return tf.reshape(generalized_degree_features, [-1, 1])


def compute_square_clustering(edge_ranges, node_ranges, edge_index):
    """Compute square clustering coefficient for each node."""
    square_clustering_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="Square Clustering"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)

        num_nodes = node_ranges[index + 1] - node_ranges[index]
        square_clustering = nx.square_clustering(G)

        # Ensure square clustering values for all nodes, including isolated ones
        square_clustering_values = [square_clustering.get(node, 0) for node in range(num_nodes)]
        square_clustering_features.extend(square_clustering_values)

    square_clustering_features = tf.convert_to_tensor(square_clustering_features, dtype=tf.float32)
    return tf.reshape(square_clustering_features, [-1, 1])


def compute_clustering_coefficient(edge_ranges, node_ranges, edge_index):
    """Compute clustering coefficient for each node."""
    clustering_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="Clustering Coeff"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.Graph()
        G.add_edges_from(graph_edges)

        num_nodes = node_ranges[index + 1] - node_ranges[index]
        clustering = nx.clustering(G)

        # Ensure clustering values for all nodes, including isolated ones
        clustering_values = [clustering.get(node, 0) for node in range(num_nodes)]
        clustering_features.extend(clustering_values)

    clustering_features = tf.convert_to_tensor(clustering_features, dtype=tf.float32)
    return tf.reshape(clustering_features, [-1, 1])

def compute_pagerank(edge_ranges, node_ranges, edge_index):
    """Compute PageRank for each node."""
    pagerank_features = []

    for index in tqdm(range(len(edge_ranges) - 1), desc="PageRank"):
        edge_start = edge_ranges[index]
        edge_end = edge_ranges[index + 1]
        graph_edges = edge_index[edge_start:edge_end].numpy()

        G = nx.DiGraph()
        G.add_edges_from(graph_edges)

        num_nodes = node_ranges[index + 1] - node_ranges[index]
        pr = nx.pagerank(G)

        # Ensure pagerank values for all nodes, including isolated ones
        pagerank = [pr.get(node, 0) for node in range(num_nodes)]
        pagerank_features.extend(pagerank)

    pagerank_features = tf.convert_to_tensor(pagerank_features, dtype=tf.float32)
    return tf.reshape(pagerank_features, [-1, 1])


def compute_node_degree_oddness(edge_index, node_feat):
    all_edges = tf.reshape(edge_index, [-1])
    num_nodes = tf.shape(node_feat)[0]

    # Initialize degree tensor with zeros for all nodes (representing evenness)
    degree = tf.zeros([num_nodes], dtype=tf.int32)

    # Update degree for nodes present in edges
    degree += tf.math.unsorted_segment_sum(
        tf.ones_like(all_edges, dtype=tf.int32), all_edges, num_nodes)

    # Calculate oddness (1 if odd degree, 0 if even)
    # Since isolated nodes (degree 0) are even, they will remain 0
    oddness = tf.cast(degree % 2 == 1, tf.float32)

    # Reshape to the desired format (n, 1) where n is the number of nodes
    oddness_feature = tf.reshape(oddness, [-1, 1])
    return oddness_feature


