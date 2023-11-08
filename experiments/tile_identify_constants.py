import numpy as np
import os
from tqdm import tqdm
import glob


def find_constant_features(npz_files):
    max_node_feat = None
    min_node_feat = None
    max_config_feat = None
    min_config_feat = None

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

            # For config_feat, update the max and min values across all configs
            config_feat = data['config_feat']
            if max_config_feat is None:
                max_config_feat = config_feat.max(axis=0)
                min_config_feat = config_feat.min(axis=0)
            else:
                max_config_feat = np.maximum(max_config_feat, config_feat.max(axis=0))
                min_config_feat = np.minimum(min_config_feat, config_feat.min(axis=0))

    # Identify constant features as those where the max and min are the same
    constant_node_feat = max_node_feat == min_node_feat
    constant_config_feat = max_config_feat == min_config_feat

    return constant_node_feat, constant_config_feat

def remove_features_from_file(npz_file, output_dir, constant_node_feat, constant_config_feat):
    with np.load(npz_file, allow_pickle=True) as data:
        # Only keep non-constant features
        node_feat = data["node_feat"][:, ~constant_node_feat]
        config_feat = data["config_feat"][:, ~constant_config_feat]
        modified_data = {
            k: v for k, v in data.items() if k not in ["node_feat", "config_feat"]
        }
        modified_data["node_feat"] = node_feat
        modified_data["config_feat"] = config_feat
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
    problems = ["tile"]
    collections = ["xla"]
    configs = ["random", "default"]
    splits = ["train", "test", "valid"]

    for prob in problems:
        print(f"problem: {prob}")
        for collection in collections:
            print(f"coll: {collection}")
            #for config in configs:
                #print(f"conf: {config}")

            root = "/home/paperspace/data"
            root = "/Users/thomasrialan/data"
            input_dir = f"{root}/tpugraphs/npz/{prob}/{collection}"

            npz_train_files = glob.glob(os.path.join(input_dir, "train/*.npz"))
            npz_valid_files = glob.glob(os.path.join(input_dir, "valid/*.npz"))
            npz_test_files = glob.glob(os.path.join(input_dir, "test/*.npz"))

            npz_files = npz_train_files
            npz_files.extend(npz_valid_files)
            npz_files.extend(npz_test_files)

            constant_node_feat, constant_node_config_feat = find_constant_features(npz_files)

            for split in splits:
                input_dir = f"{root}/tpugraphs/npz/{prob}/{collection}/{split}"
                output_dir = f"{root}/clean_tiles_tpugraphs/npz/{prob}/{collection}/{split}"

                npz_files = glob.glob(os.path.join(input_dir, "*.npz"))

                remove_features_from_files(npz_files, output_dir, constant_node_feat, constant_node_config_feat)



