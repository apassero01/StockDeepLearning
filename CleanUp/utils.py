import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.utils import to_categorical
import yfinance as yf 
import plotly.graph_objects as go
import pandas_ta as pta
from tslearn.metrics import dtw
from sklearn.base import BaseEstimator, TransformerMixin


def train_test_split(dataset, tstart, tend, feature_list):
    train = dataset.loc[f"{tstart}":f"{tend}", feature_list]
    test = dataset.loc[f"{tend+1}":, feature_list]
    
    return train, test


def split_sequence(sequence, n_steps):
    X, y = [], [] 
    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence) - 6: 
            break
        
        # Extract sequence for X and y
        # seq_x = np.copy(sequence[i:end_idx,-1:])
        seq_x = sequence[i:end_idx]
        
        # Scale all features in seq_x
        # seq_x = scaler.fit_transform(seq_x)
        
        # Rescale just the last feature of seq_x
        # last_feature_scaler = MinMaxScaler()
        # seq_x[:,-1] = last_feature_scaler.fit_transform(seq_x[:,-1].reshape(-1, 1)).ravel()
        # last_feature_scaler = MinMaxScaler()
        # seq_x[:,-2] = last_feature_scaler.fit_transform(seq_x[:,-2].reshape(-1, 1)).ravel()

        # Get sequence for y
        seq_y = [sequence[end_idx][0], sequence[end_idx+1][1],
                 sequence[end_idx+2][2],sequence[end_idx+3][3],
                 sequence[end_idx+4][4],sequence[end_idx+5][5]]
        
        X.append(seq_x)
        y.append(seq_y) 

    return np.array(X), np.array(y)

def scale_patterns(seq, feature_indices):
    for index in feature_indices: 
        for ts in seq: 
            last_feature_scaler = MinMaxScaler()
            ts[:,index] = last_feature_scaler.fit_transform(np.copy(ts[:,index]).reshape(-1, 1)).ravel()
    
    return seq

def create_Cluster_Seq(seq, feature_indices):
    # Using numpy's advanced indexing to select the required features
    return seq[:, :, feature_indices]

            


def filter_clusters(labels,target_label, X,y):
    valid_indices = np.where(labels == target_label)[0]

    filtered_x_train = X[valid_indices]
    filtered_y_train = y[valid_indices]

    return filtered_x_train, filtered_y_train


    

def removeOutliers(X_train, y_train, labels):
    # Find the indices where labels are not -1
    valid_indices = np.where(labels != -1)[0]

    # Filter x_train and labels using these indices
    filtered_x_train = X_train[valid_indices]
    filtered_labels = labels[valid_indices]
    filtered_y_train = y_train[valid_indices]

    # One-hot encode the labels
    unique_clusters = len(np.unique(filtered_labels))
    one_hot_encoded_labels = to_categorical(filtered_labels, num_classes=unique_clusters)

    # Repeat the one-hot encoding across the timesteps of LSTM
    repeated_one_hot_labels = np.repeat(one_hot_encoded_labels[:, np.newaxis, :], filtered_x_train.shape[1], axis=1)

    # Append one-hot encoded labels to the X dimension of filtered_x_train
    X_train_with_labels = np.concatenate([filtered_x_train, repeated_one_hot_labels], axis=2)

    return X_train_with_labels, filtered_y_train


def quantile_pctChg_rescale(data: pd.Series, quantiles, min_max_range=[-1, 1], maintain_zero=True) -> pd.Series:
    '''
    Trim a series according to quantile values, then rescale to fall within min_max range.
    If maintain_zero is True, then 0 is maintained before and after rescaling to ensure that 0
    maintains its 0 value (useful for positive or negative percentage changes.)

    :param data: A Pandas Series
    :param quantiles: A list or tuple of 2 quantiles, such as (0.1, 0.9) to trim the data.
    :param min_max_range: A list or tuple of the min and max number to rescale the data after trimming
    :param maintain_zero: True if zero should be maintained when rescaling
    :return: a Pandas Series
    '''

    trim_vals = np.quantile(data[~np.isnan(data)], q=quantiles)
    df_temp = pd.DataFrame({"temp": data.copy()}, index=data.index)

    # Rescale around 0
    if maintain_zero:
        # Get the max quant value form either side
        trim_abs_val = np.max(np.abs(trim_vals))

        # Trim such that pos and neg values have same max quantity
        df_temp[df_temp.temp <= -trim_abs_val] = -trim_abs_val
        df_temp[df_temp.temp >= trim_abs_val] = trim_abs_val

        sel = df_temp.temp < 0
        df_temp.temp[sel] = df_temp.temp[sel] / -trim_abs_val * min_max_range[0]
        sel = df_temp.temp > 0
        df_temp.temp[sel] = df_temp.temp[sel] / trim_abs_val * min_max_range[1]

        # scaler = MinMaxScaler(feature_range=(min_max_range[0], 0))
        # df_temp.temp[sel] = scaler.fit_transform(df_temp[sel])[:,0]
        # sel = df_temp.temp > 0
        # scaler = MinMaxScaler(feature_range=(0, min_max_range[1]))
        # df_temp.temp[sel] = scaler.fit_transform(df_temp[sel])[:,0]
    else:
        # df_temp[df_temp.temp <= trim_vals[0]] = trim_vals[0]
        # df_temp[df_temp.temp >= trim_vals[1]] = trim_vals[1]
        scaler = MinMaxScaler(feature_range=min_max_range)
        df_temp.temp = scaler.fit_transform(df_temp)[:.0]

    return df_temp.temp


def rolling_sum_window(df,windows, col_string):
    for roll in windows:
            col_name = "sum" + col_string  + "_" + str(roll)
            df[col_name] = df[col_string].rolling(roll).sum()


            # df[col_name] = quantile_pctChg_rescale(df[col_name], quantiles=[.05,.95],
            #                                                       min_max_range=[-1,1])
    return df


def prepareStockDf(ticker, start):
    stockDf = yf.download(ticker,start = "2000-01-01")
    vix = yf.download('^VIX',start = '2000-01-01')
    
    stockDf['Vix'] = vix['High']

    stockDf['PctChgVix'] = stockDf['Vix'].pct_change() * 100
    stockDf['PctChgVix'].fillna(0, inplace=True)

    stockDf['PctChgClCl'] = stockDf['Close'].pct_change() * 100
    stockDf['PctChgClCl'].fillna(0, inplace=True)



    stockDf['PctChgVol'] = stockDf['Volume'].pct_change() * 100
    stockDf['PctChgVol'].fillna(0, inplace=True)

    stockDf['HighLow'] = (stockDf['High'] - stockDf['Low'])/stockDf['High'] * 100

    stockDf = rolling_sum_window(stockDf,[2,3,4,5,6],'PctChgClCl')
    stockDf['sumPctChgClCl_2'].fillna(0, inplace=True)
    stockDf['sumPctChgClCl_3'].fillna(0, inplace=True)
    stockDf['sumPctChgClCl_4'].fillna(0, inplace=True)
    stockDf['sumPctChgClCl_5'].fillna(0, inplace=True)
    stockDf['sumPctChgClCl_6'].fillna(0, inplace=True)

    stockDf = rolling_sum_window(stockDf,[2,3,4,5,6],'PctChgVol')
    stockDf['sumPctChgVol_2'].fillna(0, inplace=True)
    stockDf['sumPctChgVol_3'].fillna(0, inplace=True)
    stockDf['sumPctChgVol_4'].fillna(0, inplace=True)
    stockDf['sumPctChgVol_5'].fillna(0, inplace=True)
    stockDf['sumPctChgVol_6'].fillna(0, inplace=True)

    stockDf = rolling_sum_window(stockDf,[2,3,4,5,6],'PctChgVix')
    stockDf['sumPctChgVix_2'].fillna(0, inplace=True)
    stockDf['sumPctChgVix_3'].fillna(0, inplace=True)
    stockDf['sumPctChgVix_4'].fillna(0, inplace=True)
    stockDf['sumPctChgVix_5'].fillna(0, inplace=True)
    stockDf['sumPctChgVix_6'].fillna(0, inplace=True)

    # stockDf.loc[:, "PctChgClCl"] = quantile_pctChg_rescale(stockDf.PctChgClCl,(.1,.9),(-1,1))
    # stockDf.loc[:, "PctChgVix"] = quantile_pctChg_rescale(stockDf.PctChgVix,(.1,.9),(-1,1))
    # stockDf.loc[:, "PctChgVol"] = quantile_pctChg_rescale(stockDf.PctChgVol,(.1,.9),(-1,1))


    stockDf['rsi'] = pta.rsi(stockDf['Close'],length = 20)
    return stockDf 

def visualizeData(labels, X_train):

    n_clusters = len(set(labels))

    # Create a MinMaxScaler to scale the values of each feature between 0 and 1
    # scaler = MinMaxScaler(feature_range=(-1, 1))

    figs = []

    for idx, label in enumerate(set(labels)):
        traces = []  # Traces for the plotly figure

        # Calculate the average and standard deviation for the cluster 
        cluster_data = X_train[labels == label]
        avg_cluster = np.mean(cluster_data, axis=0)
        std_cluster = np.std(cluster_data, axis=0)

        # Compute upper and lower bounds for one standard deviation from the mean
        upper_bound = avg_cluster + std_cluster
        lower_bound = avg_cluster - std_cluster

        # Scale the data
        # scaled_avg = scaler.fit_transform(avg_cluster)
        # scaled_upper = scaler.transform(upper_bound)
        # scaled_lower = scaler.transform(lower_bound)

        x_avg = np.arange(avg_cluster.shape[0])
        for feature_idx in range(avg_cluster.shape[1]):
            y_avg = np.ones(avg_cluster.shape[0]) * feature_idx
            z_avg = avg_cluster[:, feature_idx]
            z_upper = upper_bound[:, feature_idx]
            z_lower = lower_bound[:, feature_idx]
            traces.append(go.Scatter3d(x=x_avg, y=y_avg, z=z_avg, mode='lines', line=dict(color='red', width=2)))
            traces.append(go.Scatter3d(x=x_avg, y=y_avg, z=z_upper, mode='lines', line=dict(color='green', width=1)))
            traces.append(go.Scatter3d(x=x_avg, y=y_avg, z=z_lower, mode='lines', line=dict(color='blue', width=1)))

        # Create a 3D plot for the cluster
        fig = go.Figure(data=traces)

        fig.update_layout(title="Cluster " + str(label),
                        scene=dict(xaxis_title='Time',
                                    yaxis_title='Feature Index',
                                    zaxis_title='Value'))
        
        figs.append(fig)

    # Display all the figures
    for f in figs:
        f.show()




def remove_outliers(X_cluster, X_train, y_train, labels, model, threshold_factor):
    """
    Removes outliers from X_train, y_train, and labels based on a given clustering model and threshold factor.
    
    Args:
    - X_train (array): The time series data.
    - y_train (array): The labels for the time series data.
    - labels (array): The predicted cluster assignments for each time series in X_train.
    - model (TimeSeriesKMeans): The clustering model to be used.
    - threshold_factor (float): The factor to determine the threshold for outliers.
    
    Returns:
    - X_train_filtered (array): The filtered time series data.
    - y_train_filtered (array): The filtered labels.
    - labels_filtered (array): The filtered predicted cluster assignments.
    """
    outlier_mask = np.zeros(X_cluster.shape[0], dtype=bool)  # Initialize with all points as non-outliers

    n_clusters = model.n_clusters
    for cluster_id in range(n_clusters):
        # Extract time series in this cluster and their indices
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_data = X_cluster[cluster_indices]
        
        # Compute distances to the centroid of this cluster
        centroid = model.cluster_centers_[cluster_id]
        # distances = [dtw(centroid, ts) for ts in cluster_data]
        distances = [np.linalg.norm(centroid - ts) for ts in cluster_data]
        
        # Determine threshold (using mean and standard deviation here)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold = mean_distance + threshold_factor * std_distance
        
        # Update the outlier mask for this cluster
        outlier_mask[cluster_indices] = np.array(distances) > threshold

    # Remove outliers from X_train, y_train, and labels
    X_cluster_filtered = X_cluster[~outlier_mask]
    X_train_filtered = X_train[~outlier_mask]
    y_train_filtered = y_train[~outlier_mask]
    labels_filtered = labels[~outlier_mask]

    return X_cluster_filtered, X_train_filtered, y_train_filtered, labels_filtered


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.max_abs_trimmed_ = None
    
    def fit(self, X, y=None):
        # Ensure we're working with a copy
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        # Assuming X is a DataFrame or numpy array
        low, high = np.percentile(X_copy, [5, 95], axis=0)  # axis=0 computes percentiles column-wise
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


