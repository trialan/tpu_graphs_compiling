import numpy as np
import torch
from torch_geometric.data import Data
import glob



def get_tile_data(fold="train"):
    pattern = f"/Users/thomasrialan/data/tpugraphs/npz/tile/xla/{fold}/*.npz"
    npz_files_paths = glob.glob(pattern)[:3]
    print("USING ONLY 3 FILES")
    file_datasets = [load_npz_to_data_objects(f) for f in npz_files_paths]
    dataset = [g for dataset in file_datasets for g in dataset]
    return dataset


def load_npz_to_data_objects(npz_file_path):
    data = dict(np.load(npz_file_path))
    n_configs = data['config_feat'].shape[0]
    dataset = [make_geometric_data_obj(data, ix) for ix in range(n_configs)]
    return dataset


def make_geometric_data_obj(npz_dict, i):
    conf_indep_data = extract_config_independent_data(npz_dict)
    config_features = torch.tensor(npz_dict['config_feat'][i],
                                   dtype=torch.float)
    runtime = torch.tensor([npz_dict['config_runtime'][i]],
                           dtype=torch.float)
    normalizer = torch.tensor([npz_dict['config_runtime_normalizers'][i]],
                              dtype=torch.float)
    target = runtime / normalizer
    graph_data = Data(
        x=conf_indep_data["node_feat"],
        edge_index=conf_indep_data["edge_index"],
        node_opcodes=conf_indep_data["node_opcodes"],
        config=config_features,
        y=target,
    )
    return graph_data


def extract_config_independent_data(npz_dict):
    node_opcodes = torch.tensor(npz_dict['node_opcode'],
                                dtype=torch.float)
    node_features = torch.tensor(npz_dict['node_feat'],
                                 dtype=torch.float)
    edge_index = torch.tensor(npz_dict['edge_index'],
                              dtype=torch.long)

    data = {"node_opcodes": node_opcodes,
            "node_feat": node_features,
            "edge_index": edge_index}
    return data


if __name__ == '__main__':
    d = load_npz_to_data_objects(npz_files_paths[0])

