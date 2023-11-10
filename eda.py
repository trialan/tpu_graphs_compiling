import tensorflow as tf


def standardize_features(features, mean, std, non_categorical_indices):
    mask = tf.constant([i in non_categorical_indices for i in range(tf.shape(features)[-1])], dtype=tf.bool)

    def standardize_non_categorical(x):
        return (x - mean) / std if x.dtype != tf.bool else x

    features = tf.dynamic_partition(features, tf.cast(mask, tf.int32), 2)
    features[1] = standardize_non_categorical(features[1])
    return tf.dynamic_stitch([tf.range(tf.shape(mask)[0]), tf.range(tf.shape(mask)[0])], features)


def get_op_node_features(sample):
    # Assuming the first element of the tuple is the GraphTensor
    graph_tensor = sample[0]  # This gets the GraphTensor out of the tuple
    op_node_set = graph_tensor.node_sets['op']
    op_node_features = op_node_set.features['feats']
    import pdb;pdb.set_trace() 
    return op_node_features


def analyse_node_features(node_features):
    # Here you would calculate and store the statistics you're interested in
    # For example, you could return the mean and standard deviation
    mean = tf.reduce_mean(node_features, axis=0)
    std = tf.math.reduce_std(node_features, axis=0)
    return mean, std


def analyse(train_ds):
    # Initialize accumulators for statistics
    means = []
    stds = []

    for sample in train_ds:  # This will go through all the samples in the dataset
        op_node_features = get_op_node_features(sample)
        print(f"Num. Op. Node Feats: {op_node_features.shape}")
        mean, std = analyse_node_features(op_node_features)
        means.append(mean)
        stds.append(std)

    overall_mean = tf.reduce_mean(tf.stack(means), axis=0)
    overall_std = tf.reduce_mean(tf.stack(stds), axis=0)

    print("Overall mean of node features:", overall_mean.numpy())
    print("Overall standard deviation of node features:", overall_std.numpy())
