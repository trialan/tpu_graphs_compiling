import numpy as np
import argparse
import os
import tqdm


def process_npz_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

    for npz_file in tqdm.tqdm(npz_files, desc="Processing NPZ files"):
        npz_data = np.load(os.path.join(input_dir, npz_file), allow_pickle=True)
        modified_npz_data = remove_constant_features(npz_data)
        output_file_path = os.path.join(output_dir, npz_file)
        np.savez_compressed(output_file_path, **modified_npz_data)


def remove_constant_features(npz_data):
    node_feat = npz_data['node_feat']
    node_config_feat = npz_data['node_config_feat']
    constant_config_features = np.all(node_config_feat == -1, axis=(0, 1))
    node_config_feat = node_config_feat[:, :, ~constant_config_features]
    constant_node_features = np.all(node_feat == node_feat[0, :], axis=0)
    node_feat = node_feat[:, ~constant_node_features]
    modified_data = {k: v for k, v in npz_data.items() if k not in ['node_feat', 'node_config_feat']}
    modified_data['node_feat'] = node_feat
    modified_data['node_config_feat'] = node_config_feat
    return modified_data


def main():
    parser = argparse.ArgumentParser(description='Remove redundant features from NPZ dataset.')
    parser.add_argument('--collection', type=str, choices=['xla', 'nlp'], required=True,
                        help='The collection of the dataset, either "xla" or "nlp".')
    parser.add_argument('--type', type=str, choices=['random', 'default'], required=True,
                        help='The type of the dataset, either "random" or "default".')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'valid'], required=True,
                        help='The split of the dataset, either "train", "test", or "valid".')
    args = parser.parse_args()
    input_directory = "/home/paperspace/data/test"
    output_directory = "/home/paperspace/data/clean_test"
    #input_directory = f'/home/paperspace/data/tpugraphs/npz/layout/{args.collection}/{args.type}/{args.split}'
    #output_directory = f'/home/paperspace/data/clean_tpugraphs_v1/npz/layout/{args.collection}/{args.type}/{args.split}'

    process_npz_directory(input_directory, output_directory)

if __name__ == "__main__":
    main()

