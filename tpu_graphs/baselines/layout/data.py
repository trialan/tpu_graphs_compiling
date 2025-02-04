import collections
import warnings
import functools
import hashlib
import pickle
import io
import os
from typing import NamedTuple

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tqdm
from scipy.stats import yeojohnson, yeojohnson_normmax
from sklearn.preprocessing import RobustScaler, PowerTransformer

from tpu_graphs.baselines.layout.features import (
        compute_pagerank,
        compute_node_degree_oddness,
        compute_clustering_coefficient,
        compute_square_clustering, #a bit slow
        compute_generalized_degree,
        compute_structural_holes_metrics,
        compute_degree_centrality,
        compute_in_degree_centrality,
        compute_out_degree_centrality,
        compute_hits,
        compute_average_neighbor_degree,
        )

_TOY_DATA = flags.DEFINE_bool(
    "toy_data",
    False,
    "If set, then only 5 examples will be used in each of "
    "{train, test, validation} partitions.",
)

def print_nans(x):
    x = tf.math.is_nan(x)
    print(f"NaNs: {tf.reduce_sum(tf.cast(x, tf.int32)).numpy()}")


class LayoutExample(NamedTuple):
    """Single example of layout graph."""
    total_nodes: tf.Tensor  # shape []
    total_edges: tf.Tensor  # shape []
    total_configs: tf.Tensor  # shape []
    total_config_nodes: tf.Tensor  # shape []

    node_features: tf.Tensor  # shape [total_nodes, node_feat_size]
    node_ops: tf.Tensor  # shape [total_nodes]
    edges: tf.Tensor  # shape [total_edges, 2]
    # shape[total_configs, total_config_nodes, conf_feat_size]:
    node_config_features: tf.Tensor
    config_runtimes: tf.Tensor  # shape [total_configs]
    argsort_config_runtimes: tf.Tensor  # shape [total_configs]
    graph_id: tf.Tensor  # shape []

    node_config_ids: tf.Tensor  # shape [total_config_nodes]
    node_splits: tf.Tensor

    def to_graph_tensor(
        self, config_samples: int = -1, max_nodes: int = -1
        ) -> tfgnn.GraphTensor:
        config_features = self.node_config_features
        config_runtimes = self.config_runtimes
        num_config_nodes = tf.shape(config_features)[1]
        config_node_ids = tf.range(num_config_nodes, dtype=tf.int32)

        # If sampling is requested.
        if config_samples >= 0:
            argsort_config_runtimes = self.argsort_config_runtimes
            input_num_configs = tf.shape(self.config_runtimes)[0]
            # Skew sampling towards good runtimes.
            select_idx = tf.nn.top_k(
                # Sample wrt GumbulSoftmax([NumConfs, NumConfs-1, ..., 1])
                tf.cast(
                    (input_num_configs - tf.range(input_num_configs))
                    / input_num_configs,
                    tf.float32,
                )
                - tf.math.log(
                    -tf.math.log(tf.random.uniform([input_num_configs], 0, 1))
                ),
                config_samples,
            )[1]

            select_idx = tf.gather(argsort_config_runtimes, select_idx)
            # num_configs = config_samples
            config_runtimes = tf.gather(config_runtimes, select_idx)
            config_features = tf.gather(config_features, select_idx)

        ## As we do dropout on (sampled) nodes, maintain a list of edges to keep.
        keep_feed_src = full_feed_src = self.edges[:, 0]
        keep_feed_tgt = full_feed_tgt = self.edges[:, 1]
        keep_config_src = full_config_src = tf.range(tf.shape(self.node_config_ids)[0])
        keep_config_tgt = full_config_tgt = tf.cast(self.node_config_ids, tf.int32)
        op_node_ids = tf.range(self.total_nodes, dtype=tf.int32)
        node_is_selected = tf.ones([self.total_nodes], dtype=tf.bool)
        kept_node_ratio = tf.ones([], dtype=tf.float32)
        node_ops = self.node_ops
        node_feats = self.node_features

        if max_nodes >= 0:
            num_segments = tf.cast(tf.math.ceil(self.total_nodes / max_nodes), tf.int32)
            segment_id = tf.random.uniform(
                shape=[], minval=0, maxval=num_segments, dtype=tf.int32
            )
            start_idx = segment_id * max_nodes
            end_idx = (segment_id + 1) * max_nodes
            end_idx = tf.minimum(end_idx, self.total_nodes)
            node_is_selected = tf.logical_and(
                op_node_ids >= start_idx, op_node_ids < end_idx
            )

            feed_edge_mask = tf.logical_and(
                self.edges >= start_idx, self.edges < end_idx
            )
            feed_edge_mask = tf.logical_and(feed_edge_mask[:, 0], feed_edge_mask[:, 1])
            config_edge_mask = tf.logical_and(
                full_config_tgt >= start_idx, full_config_tgt < end_idx
            )

            kept_node_ratio = tf.cast(
                (end_idx - start_idx) / self.total_nodes, tf.float32
            )

            keep_feed_src = tf.boolean_mask(full_feed_src, feed_edge_mask)
            keep_feed_tgt = tf.boolean_mask(full_feed_tgt, feed_edge_mask)

            keep_config_src = tf.boolean_mask(full_config_src, config_edge_mask)
            keep_config_tgt = tf.boolean_mask(full_config_tgt, config_edge_mask)

        return tfgnn.GraphTensor.from_pieces(
            node_sets={
                "op": tfgnn.NodeSet.from_fields(
                    sizes=tf.shape(op_node_ids),
                    features={
                        "op": node_ops,
                        "feats": node_feats,
                        "selected": node_is_selected,
                    },
                ),
                "nconfig": tfgnn.NodeSet.from_fields(  # Node-specific configs.
                    features={
                        "feats": tf.transpose(config_features, [1, 0, 2]),
                    },
                    sizes=tf.shape(self.node_config_ids),
                ),
                "g": tfgnn.NodeSet.from_fields(
                    features={
                        "graph_id": tf.expand_dims(self.graph_id, 0),
                        "runtimes": tf.expand_dims(config_runtimes, 0),
                        "kept_node_ratio": tf.expand_dims(kept_node_ratio, 0),
                    },
                    sizes=tf.constant([1]),
                ),
            },
            edge_sets={
                "config": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(full_config_src),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("nconfig", full_config_src),
                        target=("op", full_config_tgt),
                    ),
                ),
                "feed": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(full_feed_src),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("op", full_feed_src), target=("op", full_feed_tgt)
                    ),
                ),
                "g_op": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(op_node_ids),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("g", tf.zeros_like(op_node_ids)),
                        target=("op", op_node_ids),
                    ),
                ),
                "g_config": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(config_node_ids),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("g", tf.zeros_like(config_node_ids)),
                        target=("nconfig", config_node_ids),
                    ),
                ),
                "sampled_config": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(keep_config_src),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("nconfig", keep_config_src),
                        target=("op", keep_config_tgt),
                    ),
                ),
                "sampled_feed": tfgnn.EdgeSet.from_fields(
                    sizes=tf.shape(keep_feed_src),
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=("op", keep_feed_src), target=("op", keep_feed_tgt)
                    ),
                ),
            },
        )


class NpzDatasetPartition:
    """Holds one data partition (train, test, validation) on device memory."""

    def __init__(self):
        self._data_dict: dict[str, list[np.ndarray]] = collections.defaultdict(list)
        self._num_edges: list[int] = [0]  # prepend with 0 to prep for cumsum.
        self._num_configs: list[int] = [0]  # ^^
        self._num_nodes: list[int] = [0]  # ^^
        self._num_config_nodes: list[int] = [0]  # ^^
        self._num_node_splits: list[int] = [0]  # ^^

        # Populated in `finalize()`.
        self.node_feat: "tf.Tensor | None" = None  # indexed by node_ranges.
        self.node_opcode: "tf.Tensor | None" = None  # ^^
        self.edge_index: "tf.Tensor | None" = None  # indexed by edge_ranges.
        self.config_runtime: "tf.Tensor | None" = None  # indexed by config_ranges.
        self.argsort_config_runtime: tf.Tensor | None = None  # by flat_config_ranges.
        self.graph_id: "tf.Tensor | None" = None
        # indexed by config_ranges and config_node_ranges
        self.node_config_feat: "tf.Tensor | None" = None

        # finalize() sets to: cumsum([0, numEdges(graph_1), numEdges(graph_2), ..]).
        self.edge_ranges: "tf.Tensor | None" = None
        # finalize() sets to: cumsum([0, numNodes(graph_1), numNodes(graph_2), ..]).
        self.node_ranges: "tf.Tensor | None" = None
        # finalize() sets to: cumsum([0, numConfigs(graph_1), nCfgs(graph_2), ..]).
        self.config_ranges: "tf.Tensor | None" = None
        # finalize() sets to: cumsum([0, numModules(graph_1), nModul(graph_2), ..]).
        self.config_node_ranges: "tf.Tensor | None" = None
        # _compute_flat_config_ranges (via finalize() and load_from_file()) sets to:
        # cumsum([0, numConfigs(graph_1) * numModules(graph_1), ... ])
        self.flat_config_ranges: "tf.Tensor | None" = None

        self.node_split_ranges: "tf.Tensor | None" = None
        self.node_splits: "tf.Tensor | None" = None
        self.node_config_ids: "tf.Tensor | None" = None

    def save_to_file(self, cache_file: str):
        """Saves dataset as numpy. Can be restored with `load_from_file`."""
        print("Saving ...")
        assert self.node_feat is not None, "finalize() was not invoked"
        assert self.node_opcode is not None
        assert self.edge_index is not None
        assert self.node_config_feat is not None
        assert self.config_runtime is not None
        assert self.argsort_config_runtime is not None
        assert self.node_splits is not None
        assert self.node_config_ids is not None

        assert self.graph_id is not None
        assert self.edge_ranges is not None
        assert self.node_ranges is not None
        assert self.config_ranges is not None
        assert self.config_node_ranges is not None
        assert self.node_split_ranges is not None
        assert self.flat_config_ranges is not None

        np_dict = dict(
            node_feat=self.node_feat.numpy(),
            node_opcode=self.node_opcode.numpy(),
            edge_index=self.edge_index.numpy(),
            node_config_feat=self.node_config_feat.numpy(),
            config_runtime=self.config_runtime.numpy(),
            argsort_config_runtime=self.argsort_config_runtime.numpy(),
            edge_ranges=self.edge_ranges.numpy(),
            node_ranges=self.node_ranges.numpy(),
            config_ranges=self.config_ranges.numpy(),
            node_split_ranges=self.node_split_ranges.numpy(),
            flat_config_ranges=self.flat_config_ranges.numpy(),
            config_node_ranges=self.config_node_ranges.numpy(),
            node_splits=self.node_splits.numpy(),
            node_config_ids=self.node_config_ids.numpy(),
        )
        bytes_io = io.BytesIO()
        np.savez_compressed(bytes_io, **np_dict)
        """
        with tf.io.gfile.GFile(cache_file, "wb") as fout:
            fout.write(bytes_io.getvalue())
        print("wrote " + cache_file)
        graph_ids_file = cache_file + ".graphs.txt"
        with tf.io.gfile.GFile(graph_ids_file, "w") as fout:
            fout.write(b"\n".join(self.graph_id.numpy().tolist()).decode())
        print("wrote " + graph_ids_file)
        """

    def load_from_file(self, cache_file: str):
        """Loads dataset from numpy file."""
        np_dict = np.load(tf.io.gfile.GFile(cache_file, "rb"))
        self.node_feat = tf.constant(np_dict["node_feat"])
        self.node_opcode = tf.constant(np_dict["node_opcode"])
        self.edge_index = tf.constant(np_dict["edge_index"])
        self.node_config_feat = tf.constant(np_dict["node_config_feat"])
        self.config_runtime = tf.constant(np_dict["config_runtime"])
        self.argsort_config_runtime = tf.constant(np_dict["argsort_config_runtime"])
        self.edge_ranges = tf.constant(np_dict["edge_ranges"])
        self.node_ranges = tf.constant(np_dict["node_ranges"])
        self.config_ranges = tf.constant(np_dict["config_ranges"])
        self.config_node_ranges = tf.constant(np_dict["config_node_ranges"])
        self.node_splits = tf.constant(np_dict["node_splits"])
        self.node_config_ids = tf.constant(np_dict["node_config_ids"])
        self.node_split_ranges = tf.constant(np_dict["node_split_ranges"])
        self.flat_config_ranges = tf.constant(np_dict["flat_config_ranges"])
        graph_ids = tf.io.gfile.GFile(cache_file + ".graphs.txt", "r").readlines()
        self.graph_id = tf.stack([graph_id.rstrip() for graph_id in graph_ids])
        self._compute_flat_config_ranges()
        print("loaded from " + cache_file)

    def add_npz_file(
        self,
        graph_id: str,
        npz_file: np.lib.npyio.NpzFile,
        min_configs: int = 2,
        max_configs=-1,
    ):
        npz_data = dict(npz_file.items())
        num_configs = npz_data["node_config_feat"].shape[0]
        num_config_nodes = npz_data["node_config_feat"].shape[1]
        num_nodes = npz_data["node_feat"].shape[0]
        num_edges = npz_data["edge_index"].shape[0]
        node_ranges = np.array([0, num_nodes])  # Assuming one graph per file
        edge_ranges = np.array([0, num_edges])  # Assuming one graph per file

        edge_index = npz_data['edge_index']

        avg_neigh_degree = compute_average_neighbor_degree(
                edge_ranges, node_ranges, edge_index)

        outdegree_centrality = compute_out_degree_centrality(
                edge_ranges, node_ranges, edge_index)

        indegree_centrality = compute_in_degree_centrality(
                edge_ranges, node_ranges, edge_index)

        degree_centrality = compute_degree_centrality(
                edge_ranges, node_ranges, edge_index)

        clustering_coeff = compute_clustering_coefficient(
                edge_ranges, node_ranges, edge_index)

        gen_degree = compute_generalized_degree(
                edge_ranges, node_ranges, edge_index)

        hubs, authorities = compute_hits(
                edge_ranges, node_ranges, edge_index)

        pagerank_features = compute_pagerank(
                edge_ranges, node_ranges, edge_index)

        evenness_feature = compute_node_degree_oddness(
                edge_index, num_nodes)

        npz_data['node_feat'] = tf.concat([
            npz_data['node_feat'],
            avg_neigh_degree,
            outdegree_centrality,
            indegree_centrality,
            degree_centrality,
            clustering_coeff,
            gen_degree,
            hubs,
            authorities,
            pagerank_features,
            evenness_feature,
            ], axis=-1)

        npz_data["node_splits"] = npz_data["node_splits"].reshape([-1])
        npz_data["argsort_config_runtime"] = np.argsort(npz_data["config_runtime"])
        if num_configs < min_configs:
            print("skipping graph with only %i configurations" % num_configs)
            return
        if max_configs > 0 and num_configs > max_configs:
            third = max_configs // 3
            keep_indices = np.concatenate(
                [
                    npz_data["argsort_config_runtime"][:third],  # Good configs.
                    npz_data["argsort_config_runtime"][-third:],  # Bad configs.
                    np.random.choice(
                        npz_data["argsort_config_runtime"][third:-third],
                        max_configs - 2 * third,
                    ),
                ],
                axis=0,
            )
            num_configs = max_configs
            npz_data["node_config_feat"] = npz_data["node_config_feat"][keep_indices]
            npz_data["config_runtime"] = npz_data["config_runtime"][keep_indices]
            npz_data["argsort_config_runtime"] = np.argsort(  # re-sort.
                npz_data["config_runtime"]
            )
        npz_data["node_config_feat"] = npz_data["node_config_feat"].reshape(
            (num_configs * num_config_nodes, -1)
        )
        for key, ndarray in npz_data.items():
            self._data_dict[key].append(ndarray)
        self._data_dict["graph_id"].append(np.array(graph_id))
        num_nodes = npz_data["node_feat"].shape[0]
        num_edges = npz_data["edge_index"].shape[0]


        assert num_config_nodes == npz_data["node_config_ids"].shape[0]
        assert num_nodes == npz_data["node_opcode"].shape[0]
        assert num_configs == npz_data["config_runtime"].shape[0]
        self._num_nodes.append(num_nodes)
        self._num_config_nodes.append(num_config_nodes)
        self._num_node_splits.append(npz_data["node_splits"].shape[0])
        self._num_edges.append(num_edges)
        self._num_configs.append(num_configs)

    def finalize(self):
        """Combines the list of dicts to contiguous tensors (by concat or stack).

        Afterwards, caller is able to call `get_item()` on this class instance.
        """
        self.graph_id = tf.stack(self._data_dict.pop("graph_id"), axis=0)
        self.node_feat = tf.concat(self._data_dict.pop("node_feat"), axis=0)
        self.node_opcode = tf.concat(self._data_dict.pop("node_opcode"), axis=0)
        self.edge_index = tf.concat(self._data_dict.pop("edge_index"), axis=0)

        self.node_config_feat = tf.concat(
            self._data_dict.pop("node_config_feat"), axis=0
        )
        self.config_runtime = tf.concat(self._data_dict.pop("config_runtime"), axis=0)
        self.argsort_config_runtime = tf.concat(
            self._data_dict.pop("argsort_config_runtime"), axis=0
        )
        self.node_config_ids = tf.concat(self._data_dict.pop("node_config_ids"), axis=0)
        self.node_splits = tf.concat(self._data_dict.pop("node_splits"), axis=0)

        self.edge_ranges = tf.cumsum(self._num_edges)
        self.node_ranges = tf.cumsum(self._num_nodes)
        self.config_node_ranges = tf.cumsum(self._num_config_nodes)
        self.config_ranges = tf.cumsum(self._num_configs)
        self.node_split_ranges = tf.cumsum(self._num_node_splits)
        self._compute_flat_config_ranges()


    def _compute_flat_config_ranges(self):
        num_configs = tf.cast(  # undo cumsum.
            self.config_ranges[1:] - self.config_ranges[:-1], tf.int64
        )
        num_config_nodes = tf.cast(  # undo cumsum.
            self.config_node_ranges[1:] - self.config_node_ranges[:-1], tf.int64
        )
        self.flat_config_ranges = tf.cumsum(
            tf.concat(
                [tf.zeros([1], dtype=tf.int64), num_configs * num_config_nodes], axis=0
            )
        )

    #@tf.function(autograph=False)
    def get_item(self, index: int) -> LayoutExample:
        node_start = self.node_ranges[index]
        node_end = self.node_ranges[index + 1]
        edge_start = self.edge_ranges[index]
        edge_end = self.edge_ranges[index + 1]
        config_start = self.config_ranges[index]
        config_end = self.config_ranges[index + 1]
        config_node_start = self.config_node_ranges[index]
        config_node_end = self.config_node_ranges[index + 1]
        flat_config_start = self.flat_config_ranges[index]
        flat_config_end = self.flat_config_ranges[index + 1]
        node_split_start = self.node_split_ranges[index]
        node_split_end = self.node_split_ranges[index + 1]

        num_configs = config_end - config_start
        num_config_nodes = config_node_end - config_node_start

        flat_config = self.node_config_feat[flat_config_start:flat_config_end]
        config_tensor = tf.reshape(
            flat_config, [num_configs, num_config_nodes, flat_config.shape[-1]]
        )

        return LayoutExample(
            node_features=self.node_feat[node_start:node_end],
            node_ops=self.node_opcode[node_start:node_end],
            edges=tf.cast(self.edge_index[edge_start:edge_end], tf.int32),
            node_config_features=config_tensor,
            node_config_ids=tf.cast(
                self.node_config_ids[config_node_start:config_node_end], tf.int32
            ),
            node_splits=self.node_splits[node_split_start:node_split_end],
            config_runtimes=self.config_runtime[config_start:config_end],
            argsort_config_runtimes=(
                self.argsort_config_runtime[config_start:config_end]
            ),
            graph_id=self.graph_id[index],
            total_nodes=node_end - node_start,
            total_edges=edge_end - edge_start,
            total_configs=config_end - config_start,
            total_config_nodes=config_node_end - config_node_start,
        )

    def get_graph_tensors_dataset(
        self, config_samples: int, max_nodes: int = -1
    ) -> tf.data.Dataset:
        if self.edge_ranges is None:
            raise ValueError("finalize() was not invoked.")
        dataset = tf.data.Dataset.range(self.edge_ranges.shape[0] - 1)
        dataset = dataset.map(self.get_item, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            functools.partial(
                LayoutExample.to_graph_tensor,
                config_samples=config_samples,
                max_nodes=max_nodes,
            )
        )
        return dataset

    def iter_graph_tensors(self):
        if self.edge_ranges is None:
            raise ValueError("finalize() was not invoked.")
        assert self.edge_ranges is not None
        for i in range(self.edge_ranges.shape[0] - 1):
            yield self.get_item(i).to_graph_tensor()


def compute_pca_components(tensor):
    print("Computing tensor PCA")
    covariance_matrix = tf.matmul(tensor, tensor, transpose_a=True) / tf.cast(tf.shape(tensor)[0], tf.float32)
    S, U, V = tf.linalg.svd(covariance_matrix, compute_uv=True)
    eigenvector_matrix = U
    return eigenvector_matrix


class NpzDataset(NamedTuple):
    """Contains all partitions of the dataset."""

    train: NpzDatasetPartition
    validation: NpzDatasetPartition
    test: NpzDatasetPartition

    @property
    def num_ops(self):
        return (
            int(
                tf.reduce_max(
                    [
                        tf.reduce_max(self.train.node_opcode),
                        tf.reduce_max(self.validation.node_opcode),
                        tf.reduce_max(self.test.node_opcode),
                    ]
                ).numpy()
            )
            + 1
        )
    def _get_normalizer(self, tensor):
        mean, variance = tf.nn.moments(tensor, axes=[0])
        columns_to_keep = tf.reduce_max(tensor, axis=0) != tf.reduce_min(tensor, axis=0)
        print(f"Keeping {sum(columns_to_keep.numpy())} columns")
        masked_tensor = tf.boolean_mask(tensor, columns_to_keep, axis=1)
        train_mean, train_variance = tf.nn.moments(masked_tensor, axes=[0])
        eigenvector_matrix = compute_pca_components(masked_tensor)
        return columns_to_keep, train_mean, train_variance, eigenvector_matrix

    def _apply_normalizer(self, feature_matrix, used_columns, mean, variance, eigenvector_matrix):
        feature_matrix = tf.boolean_mask(feature_matrix, used_columns, axis=1)
        feature_matrix_standardized = (feature_matrix - mean) / tf.sqrt(variance)
        decorrelated_feature_matrix = tf.matmul(feature_matrix_standardized, eigenvector_matrix)
        print_nans(decorrelated_feature_matrix)
        return decorrelated_feature_matrix

    def normalize(self, max_configs):
        print("Getting node normalizer")
        normalizer_args = self._get_normalizer(self.train.node_feat)
        print("Normalizing train")
        self.train.node_feat = self._apply_normalizer(
            self.train.node_feat, *normalizer_args
        )
        print("Normalizing valid")
        self.validation.node_feat = self._apply_normalizer(
            self.validation.node_feat, *normalizer_args
        )
        print("Normalizing test")
        self.test.node_feat = self._apply_normalizer(
            self.test.node_feat, *normalizer_args
        )

        print("Getting config normalizer")
        normalizer_args = self._get_normalizer(self.train.node_config_feat)
        print("Normalizing config train")
        self.train.node_config_feat = self._apply_normalizer(
            self.train.node_config_feat, *normalizer_args
        )
        print("Normalizing config valid")
        self.validation.node_config_feat = self._apply_normalizer(
            self.validation.node_config_feat, *normalizer_args
        )
        print("Normalizing config test")
        self.test.node_config_feat = self._apply_normalizer(
            self.test.node_config_feat, *normalizer_args
        )


def scale_cols(m, col_ixs):
    m = m.numpy()
    for ix in col_ixs:
        minix = min(m[:, ix])
        maxix = max(m[:, ix])
        m[:,ix] = (m[:,ix] - minix) / (maxix - minix)
    m = tf.convert_to_tensor(m, dtype=tf.float32)
    return m


def savepickle(m, filename):
    with open(filename, "wb") as f:
        pickle.dump(m, f)


def loadpickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def plot_outliers(feature_matrix):
    num_features = feature_matrix.shape[1]

    # Creating a box plot for each feature
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=feature_matrix, orient="h", palette="Set2")
    plt.title("Box Plot for Each Feature")
    plt.xlabel("Values")
    plt.ylabel("Features")
    plt.show()

def get_npz_split(
    split_path: str, min_configs=2, max_configs=-1, cache_dir=None
) -> NpzDatasetPartition:
    """Returns data for a single partition."""
    glob_pattern = os.path.join(split_path, "*.npz")
    files = tf.io.gfile.glob(glob_pattern)
    #files = sorted(tf.io.gfile.glob(glob_pattern))[:3]

    if not files:
        raise ValueError("No files matched: " + glob_pattern)

    npz_dataset = NpzDatasetPartition()
    for filename in tqdm.tqdm(files, desc="adding npz files (+features, runtime norm)"):
        np_data = np.load(tf.io.gfile.GFile(filename, "rb"))
        graph_id = os.path.splitext(os.path.basename(filename))[0]
        npz_dataset.add_npz_file(
            graph_id, np_data, min_configs=min_configs, max_configs=max_configs
        )
    npz_dataset.finalize()
    return npz_dataset


def get_npz_dataset(
    root_path: str,
    min_train_configs=-1,
    max_train_configs=-1,
    cache_dir: "None | str" = None,
) -> NpzDataset:
    npz_dataset = NpzDataset(
        train=get_npz_split(
            os.path.join(root_path, "train"),
            cache_dir=cache_dir,
            min_configs=min_train_configs,
            max_configs=max_train_configs,
        ),
        validation=get_npz_split(
            os.path.join(root_path, "valid"),
            cache_dir=cache_dir,
            min_configs=min_train_configs,
            max_configs=max_train_configs,
        ),
        test=get_npz_split(os.path.join(root_path, "test"), cache_dir=cache_dir),
    )
    npz_dataset.normalize(max_train_configs)
    return npz_dataset

