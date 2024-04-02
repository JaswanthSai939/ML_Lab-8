import numpy as np
import pandas as pd

def entropy(labels):
    """Calculate the entropy of a set of labels."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def information_gain(data, feature_name, target_name):
    """Calculate the information gain for a given feature."""
    # Calculate total entropy before splitting
    total_entropy = entropy(data[target_name])
    
    # Calculate the weighted entropy after splitting on the feature
    weighted_entropy = 0
    unique_values = data[feature_name].unique()
    for value in unique_values:
        subset = data[data[feature_name] == value]
        subset_entropy = entropy(subset[target_name])
        weight = len(subset) / len(data)
        weighted_entropy += weight * subset_entropy
    
    # Calculate information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_root_feature(data, target_name):
    """Find the feature with the highest information gain."""
    best_gain = 0
    best_feature = None
    for feature in data.columns:
        if feature != target_name:  # Exclude target column
            gain = information_gain(data, feature, target_name)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
    return best_feature

# Load data from CSV
file_path = r'C:\Users\sai jaswanth\Downloads\Parkinsson disease (1).csv'
data = pd.read_csv(file_path)

# Assuming the target column is 'status', change it to your target column name if different
target_name = 'status'

# Find the root feature
root_feature = find_root_feature(data, target_name)
print("Root feature:", root_feature)
