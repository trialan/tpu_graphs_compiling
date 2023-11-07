import numpy as np
import os
from tqdm import tqdm
import glob


def find_constant_features(npz_files):
    max_node_feat = None
    min_node_feat = None
    max_node_config_feat = None
    min_node_config_feat = None

    # Iterate over each file to update the max and min for each feature
    for npz_file in tqdm(npz_files, desc="Identifying constant features"):
        with np.load(npz_file, allow_pickle=True) as data:
            # For node_feat, update the max and min values across all nodes
            node_feat = data['node_feat']
            if max_node_feat is None:
                max_node_feat = node_feat.max(axis=0)
                min_node_feat = node_feat.min(axis=0)
            else:
                max_node_feat = np.maximum(max_node_feat, node_feat.max(axis=0))
                min_node_feat = np.minimum(min_node_feat, node_feat.min(axis=0))

            # For node_config_feat, update the max and min values across all configs
            node_config_feat = data['node_config_feat']
            if max_node_config_feat is None:
                max_node_config_feat = node_config_feat.max(axis=(0, 1))
                min_node_config_feat = node_config_feat.min(axis=(0, 1))
            else:
                max_node_config_feat = np.maximum(max_node_config_feat, node_config_feat.max(axis=(0, 1)))
                min_node_config_feat = np.minimum(min_node_config_feat, node_config_feat.min(axis=(0, 1)))

    # Identify constant features as those where the max and min are the same
    constant_node_feat = max_node_feat == min_node_feat
    constant_node_config_feat = (max_node_config_feat == min_node_config_feat) & (min_node_config_feat == -1)

    return constant_node_feat, constant_node_config_feat


def remove_features_from_file(
    npz_file, output_dir, constant_node_feat, constant_node_config_feat
):
    with np.load(npz_file, allow_pickle=True) as data:
        node_feat = data["node_feat"][:, ~constant_node_feat]
        node_config_feat = data["node_config_feat"][:, :, ~constant_node_config_feat]
        modified_data = {
            k: v for k, v in data.items() if k not in ["node_feat", "node_config_feat"]
        }
        modified_data["node_feat"] = node_feat
        modified_data["node_config_feat"] = node_config_feat
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, os.path.basename(npz_file))
        np.savez_compressed(output_file_path, **modified_data)


def remove_features_from_files(
    npz_files, output_dir, constant_node_feat, constant_node_config_feat
):
    for npz_file in tqdm(npz_files, desc="Removing constant features"):
        remove_features_from_file(
            npz_file, output_dir, constant_node_feat, constant_node_config_feat
        )


if __name__ == '__main__':
    problems = ["layout", "tile"]
    collections = ["xla", "nlp"]
    configs = ["random", "default"]
    splits = ['train', 'test', 'valid']

    for prob in problems:
        for collection in collections:
            for config in configs:
                for split in splits:
                    print(f"coll: {collection}")
                    print(f"conf: {config}")
                    print(f"split: {split}")

                    input_dir = f"/users/thomasrialan/data/tpugraphs/npz/{prob}/{collection}/{config}/{split}"
                    output_dir = f"/users/thomasrialan/data/clean_tpugraphs_v2/npz/{prob}/{collection}/{config}/{split}"

                    npz_files = glob.glob(os.path.join(input_dir, "*.npz"))
                    constant_node_feat, constant_node_config_feat = find_constant_features(npz_files)

                    print(constant_node_config_feat)
                    print(constant_node_feat)

                    remove_features_from_files(npz_files, output_dir, constant_node_feat, constant_node_config_feat)
