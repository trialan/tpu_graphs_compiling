import numpy as np

root = "/Users/thomasrialan/data/tpugraphs/npz/layout/nlp/random/train/"
# Load the .npz file
data_path = f'cleaned_alexnet.npz'
data = np.load(data_path)

# Function to analyze the proportion of zero values in node features
def analyze_zeros(node_features):
    zero_proportions = np.mean(node_features == 0, axis=0)
    return zero_proportions

# Function to analyze the proportion of -1 values in node configuration features
def analyze_minus_ones(config_features):
    minus_one_proportions = np.mean(config_features == -1, axis=(0, 1))
    return minus_one_proportions

# Function to remove constant features
def remove_constant_features(features, threshold=1.0):
    # Calculate the variance for each feature
    variances = np.var(features, axis=0)
    # Features with variance less than the threshold are considered constant
    non_constant_indices = np.where(variances < threshold)[0]
    # Remove the constant features
    return features[:, non_constant_indices], non_constant_indices

# Perform analysis
node_feat = data['node_feat']
node_config_feat = data['node_config_feat']

zero_proportions = analyze_zeros(node_feat)
minus_one_proportions = analyze_minus_ones(node_config_feat)

# Identify and remove constant features from node_feat
cleaned_node_feat, non_constant_indices = remove_constant_features(node_feat)

# Count the number of constant features
num_constant_features = node_feat.shape[1] - len(non_constant_indices)

print(f"Number of constant features removed: {num_constant_features}")
print(f"Shape of node features before cleaning: {node_feat.shape}")
print(f"Shape of node features after cleaning: {cleaned_node_feat.shape}")
