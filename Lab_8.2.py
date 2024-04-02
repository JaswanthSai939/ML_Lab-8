import numpy as np
import pandas as pd

def equal_width_binning(data, feature_name, num_bins):
    """Perform equal width binning on a continuous-valued feature."""
    min_value = data[feature_name].min()
    max_value = data[feature_name].max()
    bin_width = (max_value - min_value) / num_bins
    
    bins = np.arange(min_value, max_value + bin_width, bin_width)
    labels = [f'bin_{i}' for i in range(1, num_bins + 1)]
    
    binned_feature = pd.cut(data[feature_name], bins=bins, labels=labels, include_lowest=True)
    return binned_feature

def frequency_binning(data, feature_name, num_bins):
    """Perform frequency binning on a continuous-valued feature."""
    bins = data[feature_name].value_counts(bins=num_bins).index.sort_values()
    labels = [f'bin_{i}' for i in range(1, num_bins + 1)]
    
    binned_feature = pd.cut(data[feature_name], bins=bins, labels=labels, include_lowest=True)
    return binned_feature

def bin_continuous_feature(data, feature_name, num_bins=None, binning_type='equal_width'):
    """Bin a continuous-valued feature."""
    if num_bins is None:
        num_bins = 5  # Default number of bins
    
    if binning_type == 'equal_width':
        try:
            return equal_width_binning(data, feature_name, num_bins)
        except KeyError:
            print(f"Error: {feature_name} column does not exist in the dataset.")
            return None
    elif binning_type == 'frequency':
        try:
            return frequency_binning(data, feature_name, num_bins)
        except KeyError:
            print(f"Error: {feature_name} column does not exist in the dataset.")
            return None
    else:
        print("Invalid binning_type. Choose either 'equal_width' or 'frequency'.")
        return None

# Load data from CSV
file_path = r'C:\Users\sai jaswanth\Downloads\Parkinsson disease (1).csv'
data = pd.read_csv(file_path)

# Example usage:
# Binning the 'A1' feature using equal width with default parameters
binned_feature = bin_continuous_feature(data, 'A1')
if binned_feature is not None:
    print("Binned feature using equal width:")
    print(binned_feature.head())

# Binning the 'A1' feature using frequency binning with custom number of bins
binned_feature_freq = bin_continuous_feature(data, 'A1', num_bins=3, binning_type='frequency')
if binned_feature_freq is not None:
    print("\nBinned feature using frequency binning:")
    print(binned_feature_freq.head())