import io
import os

import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def process_npz_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    npz_files = glob.glob(os.path.join(input_dir, '*.npz'))

    for npz_file_path in tqdm(npz_files):
        node_feat_updated, data = add_degree_features(npz_file_path)
        updated_data = {key: data[key] for key in data.files if key != 'node_feat'}
        updated_data['node_feat'] = node_feat_updated
        output_file_path = os.path.join(output_dir, os.path.basename(npz_file_path))
        bytes_io = io.BytesIO()
        np.savez_compressed(bytes_io, **updated_data)
        import pdb;pdb.set_trace() 
        with tf.io.gfile.GFile(cache_file, "wb") as fout:
            fout.write(bytes_io.getvalue())


def add_degree_features(npz_file_path):
    data = np.load(npz_file_path)
    edge_index = data['edge_index']
    node_feat = data['node_feat']

    node_degrees = np.zeros(node_feat.shape[0], dtype=int)
    for edge in edge_index:
        node_degrees[edge[0]] += 1
        node_degrees[edge[1]] += 1

    odd_even_feature = (node_degrees % 2).astype(np.float32)

    odd_even_feature = odd_even_feature.reshape(-1, 1)
    node_degrees = node_degrees.reshape(-1, 1).astype(np.float32)

    node_feat_updated = np.hstack((node_feat, node_degrees, odd_even_feature))
    return node_feat_updated, data


if __name__ == '__main__':
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
                input_dir = f"{root}/tpugraphs/npz/layout/{collection}/{config}/{split}"
                output_dir = f"{root}/nodefeats_tpugraphs/npz/layout/{collection}/{config}/{split}"

                npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
                print(f"Num. files: {len(npz_files)}")

                process_npz_files(input_dir, output_dir)

