import numpy as np
import networkx as nx
import os
import glob
from tqdm import tqdm
import io


def compute_graph_features(edge_index):
    G = nx.DiGraph()
    G.add_edges_from(edge_index)

    # Compute features and convert them to numpy arrays
    pagerank_array = np.array(list(nx.pagerank(G).values()))
    load_centrality_array = np.array(list(nx.load_centrality(G).values()))
    katz_centrality_array = np.array(list(nx.katz_centrality_numpy(G).values()))
    harmonic_centrality_array = np.array(list(nx.harmonic_centrality(G).values()))
    average_neighbor_degree_array = np.array(list(nx.average_neighbor_degree(G).values()))
    kcore_array = np.array(list(nx.core_number(G).values()))
    clustering_array = np.array(list(nx.clustering(G).values()))
    constraint_array = np.array(list(nx.constraint(G).values()))
    assortativity = nx.degree_assortativity_coefficient(G)
    assortativity_array = np.full(G.number_of_nodes(), assortativity)

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    clustering_coefficients = nx.clustering(G.to_undirected())

    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G)
        print("eigencentrality calculation didn't converge")
    except nx.PowerIterationFailedConvergence as e:
        eigenvector_centrality = {n: 0 for n in G.nodes()}
    # Create the dictionary of features
    graph_features = {
        'pagerank': pagerank_array,
        'load_centrality': load_centrality_array,
        'katz_centrality': katz_centrality_array,
        'harmonic_centrality': harmonic_centrality_array,
        'average_neighbor_degree': average_neighbor_degree_array,
        'kcore': kcore_array,
        'clustering': clustering_array,
        'constraint': constraint_array,
        'assortativity': assortativity_array,
        'in_degree': in_degrees,
        'out_degree': out_degrees,
        'clustering_coefficient': clustering_coefficients,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'eigenvector_centrality': eigenvector_centrality
    }

    return graph_features


def process_files_and_append_graph_features(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = glob.glob(os.path.join(input_dir, '*.npz'))

    for npz_file_path in tqdm(npz_files):
        data = dict(np.load(npz_file_path, allow_pickle=True))
        node_feat_original = data['node_feat']

        graph_feats = compute_graph_features(data['edge_index'])

        graph_feats_array = np.array([graph_feats[node_id] for node_id in range(len(node_feat_original))])

        node_feat_combined = np.hstack((node_feat_original, graph_feats_array))

        data['node_feat'] = node_feat_combined

        output_file_path = os.path.join(output_dir, os.path.basename(npz_file_path))

        bytes_io = io.BytesIO()
        np.savez_compressed(bytes_io, **data)
        with tf.io.gfile.GFile(output_file_path, "wb") as fout:
            fout.write(bytes_io.getvalue())


if __name__ == '__main__':
    if False:
        file = "alexnet_train_batch_32.npz"
        import time
        t0 = time.time()
        feats = compute_graph_features(dict(np.load(file))['edge_index'])
        t1 = time.time()
        print(t1-t0)

    collections = ["xla", "nlp"]
    configs = ["random", "default"]
    splits = ["train", "test", "valid"]

    for collection in collections:
        print(f"coll: {collection}")
        for config in configs:
            print(f"conf: {config}")

            root = "/Users/thomasrialan/data"
            root = "/home/paperspace/data"
            input_dir = f"{root}/tpugraphs/npz/layout/{collection}"

            for split in splits:
                print(f"Split: {split}")
                input_dir = f"{root}/nodefeats_tpugraphs/npz/layout/{collection}/{config}/{split}"
                output_dir = f"{root}/graphfeats_tpugraphs/npz/layout/{collection}/{config}/{split}"

                npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
                print(f"Num. files: {len(npz_files)}")

                process_files_and_append_graph_features(input_dir, output_dir)
