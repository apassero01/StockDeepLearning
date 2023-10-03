from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tslearn.preprocessing import TimeSeriesScalerMinMax
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from copy import deepcopy
from collections import Counter

class ScalingMethod(Enum):
    """
    Enum class to encapsulate the different scaling methods available

    SBS - Sequence by sequence scaling. Each sequence is scaled independently
    SBSG - Sequence by sequence scale with grouped features. Each sequence is scaled independently but the same scaling is applied to all features
    QUANT_MINMAX - Quantile min max scaling. Each feature is scaled independently using the quantile min max scaling method
    UNSCALED - No scaling is applied
    """
    QUANT_MINMAX = 1
    SBS = 2
    SBSG = 3
    UNSCALED = 4


class SequenceSet: 
    def __init__(self,X_cols, y_cols, dfs, x_feature_sets, y_feature_sets, n_steps = 10) -> None:
        self.X_cols = X_cols
        self.y_cols = y_cols
        self.dfs = [df.copy() for df in dfs]
        
        self.n_steps = n_steps

        self.training_dfs = []
        self.test_dfs = [] 

        self.x_feature_sets = x_feature_sets
        self.y_feature_sets = y_feature_sets
    
    def set_n_steps(self,n_steps):
        self.n_steps = n_steps
    
    def train_test_split(self):
        pass 
    
    def scale_seq(self):
        pass

class StockSequenceSet(SequenceSet):
    def __init__(self,X_cols, y_cols, dfs, tickers,start,end,interval,x_feature_sets,y_feature_sets,n_steps = 10) -> None:
        super().__init__(X_cols, y_cols, dfs, x_feature_sets, y_feature_sets, n_steps)
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval


    def train_test_split(self,t_start,t_end,feature_list = None):
        if t_start < self.start.year or t_end > self.end.year:
            raise ValueError("Invalid time range")
        
        if not feature_list:
            feature_list = list(self.X_cols.union(self.y_cols))

        self.train_start = t_start
        self.train_end = t_end

        for df in self.dfs:
            training_df, test_df = df_train_test_split(df, self.train_start, self.train_end, feature_list)
            self.training_dfs.append(training_df)
            test_df = test_df.iloc[self.n_steps:,:] # Remove the first n_steps rows to prevent data leakage
            self.test_dfs.append(test_df)
    
    def scale_seq(self): 
        scaler = ScalerSequencer(self.x_feature_sets,self.y_feature_sets,self.training_dfs,self.test_dfs,self.X_cols,self.y_cols,self.n_steps)
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_scaled, self.X_test_scaled, self.y_train_scaled, self.y_test_scaled = scaler.scale_and_sequence() 
        self.X_feature_dict, self.y_feature_dict = scaler.get_feature_dict()
        

class ScalerSequencer:
    """
    Class for dynamically scaling data based on the requirements of the features 
    Some scaling methods occur before sequences are created, others after.
    This method appropriately scales the data based on the requirements of the feature sets 
    and will return fully scaled data in 3D time sequence form. 
    """
    scalingMethods = ["SBS","SBSG","QUANT_MINMAX","UNSCALED"]

    def __init__(self,x_feature_sets,y_feature_sets,training_dfs, test_dfs,X_cols,y_cols, n_steps) -> None:
        self.x_feature_sets = x_feature_sets
        self.y_feature_sets = y_feature_sets

        self.test_dfs = deepcopy(test_dfs)
        self.training_dfs = deepcopy(training_dfs)

        self.X_cols = X_cols
        self.y_cols = y_cols

        self.n_steps = n_steps
  
    def scale_and_sequence(self):
        '''
        Pipeline function responsible for scaling features requiring df scaling first,
        then creating the sequence, then scaling the features requiring sequence scaling
        '''
        #First scale QUANT_MIN__MAX features in dataframe form 

        x_quant_min_max_feature_sets = [feature_set for feature_set in self.x_feature_sets if feature_set.scaling_method == ScalingMethod.QUANT_MINMAX] 
        y_quant_min_max_feature_sets = [feature_set for feature_set in self.y_feature_sets if feature_set.scaling_method == ScalingMethod.QUANT_MINMAX]       
        if len(x_quant_min_max_feature_sets) > 0:
            self.training_dfs, self.test_dfs = self.scale_quant_min_max(x_quant_min_max_feature_sets, self.training_dfs, self.test_dfs)
        if len(y_quant_min_max_feature_sets) > 0:
            self.training_dfs, self.test_dfs = self.scale_quant_min_max(y_quant_min_max_feature_sets, self.training_dfs, self.test_dfs)

        #Then create the sequence 
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_combined_sequence(self.training_dfs,self.test_dfs,self.n_steps)

        #Then scale the SBS features in sequence form
        x_sbs_feature_sets = [feature_set for feature_set in self.x_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBS.value]
        if len(x_sbs_feature_sets) > 0:
            self.X_train_scaled, self.X_test_scaled = self.scale_sbs(x_sbs_feature_sets,self.X_train,self.X_test)

        #Then scale the sbs features in sequence form
        x_sbsg_feature_sets = [feature_set for feature_set in self.x_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBSG.value]
        if len(x_sbsg_feature_sets) > 0:
            self.X_train_scaled, self.X_test_scaled = self.scale_sbsg(x_sbsg_feature_sets,self.X_train_scaled,self.X_test_scaled)

        #Then scale the y features in sequence form
        y_sbs_feature_sets = [feature_set for feature_set in self.y_feature_sets if feature_set.scaling_method.value == ScalingMethod.SBS.value]
        if len(y_sbs_feature_sets) > 0:
            self.y_train_scaled, self.y_test_scaled = self.y_scale_sbs(y_sbs_feature_sets,self.y_train,self.y_test)
        else: 
            self.y_train_scaled = self.y_train
            self.y_test_scaled = self.y_test

        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_train_scaled, self.X_test_scaled, self.y_train_scaled, self.y_test_scaled


    
    def create_combined_sequence(self,training_dfs, test_dfs, n_steps):
        '''
        Creates the sequence for each dataframe and concatenates them together
        '''


        X_train, X_test = np.empty((0, n_steps, len(self.X_cols))), np.empty((0, n_steps, len(self.X_cols)))
        y_train, y_test = np.empty((0, len(self.y_cols))), np.empty((0, len(self.y_cols)))

        for i in range(len(training_dfs)):
            training_set = training_dfs[i]
            test_set = test_dfs[i]

            X_train_i, y_train_i, self.X_feature_dict, self.y_feature_dict = create_sequence(training_set, self.X_cols, self.y_cols, n_steps)
            X_test_i, y_test_i, self.X_feature_dict, self.y_feature_dict = create_sequence(test_set, self.X_cols, self.y_cols, n_steps)

            X_train = np.concatenate((X_train, X_train_i), axis=0)
            X_test = np.concatenate((X_test, X_test_i), axis=0)
            y_train = np.concatenate((y_train, y_train_i), axis=0)
            y_test = np.concatenate((y_test, y_test_i), axis=0)

        return X_train, X_test, y_train, y_test
        
        
    def scale_quant_min_max(self,feature_sets, training_dfs, test_dfs):
        """
        Scales the features in the feature sets in dataframe form using custom MinMaxPercentileScaler
        """

        training_dfs = [df.copy() for df in training_dfs]
        test_dfs = [df.copy() for df in test_dfs]

        combined_train_df = pd.DataFrame()
        combined_test_df = pd.DataFrame()

        #Combine all the training and test dataframes into one dataframe
        for i in range(len(training_dfs)):
            combined_train_df = pd.concat([combined_train_df,training_dfs[i]],axis = 0)
            combined_test_df = pd.concat([combined_test_df,test_dfs[i]],axis = 0)
        
        #Scale the features in the feature sets
        for feature_set in feature_sets: 
            scaler = MinMaxPercentileScaler()
            scaler.fit(combined_train_df[feature_set.cols])

            # After fitting the scaler to the combined training dataframe, transform the individual dataframes
            for i in range(len(training_dfs)):
                training_set = training_dfs[i]
                test_set = test_dfs[i]


                count = Counter(feature_set.cols)
                items =  [item for item, count in count.items() if count > 1]
                if len(feature_set.cols) != len(set(feature_set.cols)):
                    print(items)
                    

                transformed_data = scaler.transform(training_set[feature_set.cols])

                training_set[feature_set.cols] = scaler.transform(training_set[feature_set.cols])
                test_set[feature_set.cols] = scaler.transform(test_set[feature_set.cols])

                training_dfs[i] = training_set
                test_dfs[i] = test_set
            feature_set.scalers.append(scaler)
        
        return training_dfs, test_dfs
    
    def scale_sbs(self,feature_sets,X_train,X_test):
        """
        Scales the features in each feature_set sequence by sequence independant of each other 
        """
        X_train = np.copy(X_train)
        X_test = np.copy(X_test)

        # Iterate through all the feature_sets requiring SBS scaling 
        for feature_set in feature_sets:
            # Extract the indices of the features in the feature set
            cols = feature_set.cols
            cols_indices = [self.X_feature_dict[col] for col in cols]

            # Iterate through each column, scaling each sequence in both the train and test set
            for index in cols_indices:
                for ts in X_train: 
                    scaler = MinMaxScaler(feature_set.range)
                    ts[:,index] = scaler.fit_transform(np.copy(ts[:,index].reshape(-1,1))).ravel()
                    feature_set.scalers.append(scaler)
                for ts in X_test:
                    scaler = MinMaxScaler(feature_set.range)
                    ts[:,index] = scaler.fit_transform(np.copy(ts[:,index].reshape(-1,1))).ravel()
                    feature_set.scalers.append(scaler)
        return X_train, X_test
    
    def scale_sbsg(self,feature_sets,X_train,X_test):
        """
        Scales the features in each feature_set sequence by sequence independant of each other 
        """
        X_train = np.copy(X_train)
        X_test = np.copy(X_test)

        for feature_set in feature_sets:
            cols = feature_set.cols
            cols_indices = [self.X_feature_dict[col] for col in cols]


            # Create an instance of time series scaler
        
            
            # Iterate over each sequence and scale the specified features
            for i in range(X_train.shape[0]):

            # for i in range(1):
                scaler = MinMaxScaler()
                # Extract the sequence
                sequence = np.copy(X_train[i])
                
                # Vertically stack the columns to be scaled
                combined_series = sequence[:, cols_indices].reshape(-1, 1)
                
                # Scale the combined series
                scaled_combined_series = scaler.fit_transform(np.copy(combined_series))
                # Split the scaled_combined_series back to the original shape and update the sequence
                sequence[:, cols_indices] = scaled_combined_series.reshape(sequence.shape[0], len(cols_indices))
                
                # Assign the scaled sequence back to the original 3D array
                X_train[i] = sequence

                feature_set.scalers.append(scaler)
            
            
            for i in range(X_test.shape[0]):
                scaler = MinMaxScaler()
                # Extract the sequence
                sequence = np.copy(X_test[i])
                
                # Vertically stack the columns to be scaled
                combined_series = sequence[:, cols_indices].reshape(-1, 1)
                # Scale the combined series
                scaled_combined_series = scaler.fit_transform(np.copy(combined_series))
                
                # Split the scaled_combined_series back to the original shape and update the sequence
                sequence[:, cols_indices] = scaled_combined_series.reshape(sequence.shape[0], len(cols_indices))
                
                # Assign the scaled sequence back to the original 3D array
                X_test[i] = sequence
            
                feature_set.scalers.append(scaler)

    
        return X_train, X_test
    
    def y_scale_sbs(self,feature_set, y_train, y_test): 
        """
        Scale the y features in each sequence with respect to each other. 
        This means if feature 1 is pctChg 1 day out and feature 2 is pctChg 5 days out, they will be on the same scale
        """

        y_train = np.copy(y_train)
        y_test = np.copy(y_test)
        
        for i,row in enumerate(y_train): 
            scaler = MinMaxPercentileScaler(feature_set.range)
            row = scaler.fit_transform(np.copy(row.reshape(-1,1))).ravel()
            y_train[i] = row 
            feature_set.scalers.append(scaler)
        for i,row in enumerate(y_test):
            scaler = MinMaxPercentileScaler(feature_set.range)
            row = scaler.fit_transform(np.copy(row.reshape(-1,1))).ravel()
            y_test[i] = row 
            feature_set.scalers.append(scaler)
        
        return y_train, y_test\
    
    def get_feature_dict(self):
        return self.X_feature_dict, self.y_feature_dict



def df_train_test_split(dataset, tstart, tend, feature_list):
    '''
    Split the dataset into train and test sets
    '''
    train = dataset.loc[f"{tstart}":f"{tend}", feature_list]
    test = dataset.loc[f"{tend+1}":, feature_list]

    return train, test


def create_sequence(df, X_cols, y_cols, n_steps):
    """
    Creates sequences of length n_steps from the dataframe df for the columns in X_cols and y_cols.
    """
    X, y = [], []
    sequence = df[list(X_cols.union(y_cols))].values

    # Get indices of X_cols and y_cols in the dataframe
    X_indices_df = [df.columns.get_loc(col) for col in X_cols]
    y_indices_df = [df.columns.get_loc(col) for col in y_cols]

    # Indices of features in the sequence (3D array)
    X_indices_seq = list(range(len(X_cols)))
    y_indices_seq = list(range(len(y_cols)))

    X_feature_dict = {col: index for col, index in zip(X_cols, X_indices_seq)}
    y_feature_dict = {col: index for col, index in zip(y_cols, y_indices_seq)}

    for i in range(len(sequence)):
        end_idx = i + n_steps
        if end_idx > len(sequence):
            break

        # Extract sequence for X
        seq_x = sequence[i:end_idx, :][:, X_indices_df]

        # Get sequence for y from the row at end_idx-1 of sequence for the columns in y_cols
        seq_y = sequence[end_idx - 1, y_indices_df]

        X.append(seq_x)
        y.append(seq_y)

    return X, y, X_feature_dict, y_feature_dict



class MinMaxPercentileScaler(BaseEstimator, TransformerMixin):
    """
    Custom transformer that clips data to the defined percentiles and scales it between -1 and 1. This is important as zero is maintained before and after scaling. This ensures
    the same number of values <, = and > zero are maintained.
    """
    def __init__(self,percentile=[1,99]):
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