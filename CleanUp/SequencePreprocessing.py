from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
import pandas as pd
import numpy as np




def create_sequence(df, X_cols, y_cols, n_steps):
    """
    Creates sequences of length n_steps from the dataframe df for the columns in X_cols and y_cols.
    """
    X, y = [], []
    sequence = df[X_cols + y_cols].values

    # Get indices of X_cols and y_cols
    X_indices = [df.columns.get_loc(col) for col in X_cols]
    y_indices = [df.columns.get_loc(col) for col in y_cols]

    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence):
            break

        # Extract sequence for X
        seq_x = sequence[i:end_idx, :][:, X_indices]

        # Get sequence for y from the row at end_idx-1 of sequence for the columns in y_cols
        seq_y = sequence[end_idx - 1, y_indices]

        X.append(seq_x)
        y.append(seq_y)

    return X, y


def train_test_split(dataset, tstart, tend, feature_list):
    '''
    Split the dataset into train and test sets
    '''
    train = dataset.loc[f"{tstart}":f"{tend}", feature_list]
    test = dataset.loc[f"{tend+1}":, feature_list]

    return train, test


def scale_together(sequences, feature_indices):
    # Create an instance of the scaler
    scaler = TimeSeriesScalerMinMax()
    
    # Iterate over each sequence and scale the specified features
    for i in range(sequences.shape[0]):
        # Extract the sequence
        sequence = sequences[i]
        
        # Vertically stack the columns to be scaled
        combined_series = sequence[:, feature_indices].reshape(-1, len(feature_indices))
        
        # Scale the combined series
        scaled_combined_series = scaler.fit_transform(combined_series)
        
        # Split the scaled_combined_series back to the original shape and update the sequence
        sequence[:, feature_indices] = scaled_combined_series.reshape(sequence.shape[0], len(feature_indices))
        
        # Assign the scaled sequence back to the original 3D array
        sequences[i] = sequence
        
    return sequences


class DynamicScaler:
    """
    Class for dynamically scaling data based on the requirements of the features 
    """
    scalingMethods = ["SBS","SBSG","QUANT_MINMAX","UNSCALED"]


class MinMaxPercentileScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer that clips data to the defined percentiles and scales it between -1 and 1. This is important as zero is maintained before and after scaling. This ensures
    the same number of values <, = and > zero are maintained.
    """
    def __init__(self,percentile=[5,95]):
        self.max_abs_trimmed_ = None
        self.percentile = percentile
    
    def fit(self, X, y=None):
        # Ensure we're working with a copy
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        # Assuming X is a DataFrame or numpy array
        low, high = np.percentile(X_copy, self.percentile, axis=0)  # axis=0 computes percentiles column-wise
        self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
        return self

    def transform(self, X):
        # Ensure we're working with a copy
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        # Clip data column-wise
        X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)

        # Scale values between -1 and 1 for each column, maintaining zero
        pos_mask = X_copy > 0
        neg_mask = X_copy < 0
        
        X_copy[pos_mask] = X_copy[pos_mask] / self.max_abs_trimmed_
        X_copy[neg_mask] = X_copy[neg_mask] / self.max_abs_trimmed_

        return X_copy
    
    def inverse_transform(self, X):
        # Ensure we're working with a copy
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        # If it's a DataFrame, get the numpy array
        if isinstance(X_copy, pd.DataFrame):
            X_copy = X_copy.values  # Convert to NumPy array

        X_copy = X_copy * self.max_abs_trimmed_

        return X_copy