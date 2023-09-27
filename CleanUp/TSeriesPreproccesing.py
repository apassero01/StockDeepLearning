import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from enum import Enum
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelBinarizer,
    RobustScaler,
)

class ScalingMethod(Enum):
    """
    Enum class to encapsulate the different scaling methods available

    SBS - Sequence by sequence scaling. Each sequence is scaled independently
    SBSG - Sequence by sequence scale with grouped features. Each sequence is scaled independently but the same scaling is applied to all features
    QUANT_MINMAX - Quantile min max scaling. Each feature is scaled independently using the quantile min max scaling method
    UNSCALED - No scaling is applied
    """
    SBS = 1
    SBSG = 2
    QUANT_MINMAX = 3
    UNSCALED = 4

class FeatureSet:
    """
    Class that encapsulates a set of features. A set of features is a subset of the columns in a Dateset object. 
    This is helpful for keeping track of different requirements for different groups of features
    """
    def __init__(self, scaling_method):
        self.cols = []
        self.scaling_method = scaling_method


class DataSet:
    """
    Class to encapsuate a dataset. The dataset is essentially a list of dataframes of the
    same shape and size with the same features.
    """
    def __init__(self):
        self.dfs = []
        self.df = pd.DataFrame()
        self.X_cols = set()
        self.y_cols = set()

        self.created_dataset = False
        self.created_features = False
        self.created_y_targets = False

    def update_total_df(self):
        """
        Update the total dataframe with all the dataframes in the dataset
        """
        self.df = pd.DataFrame()
        self.df = pd.concat(self.dfs, axis=0)
    
    def create_dataset(self):
        """
        Create the dataset from the tickers and start/end dates and interval
        """
        pass
    def create_features(self):
        """
        Create additonal featues for the dataset
        """
        pass

    def create_y_targets(self, cols_to_create_targets):
        """
        Create target y values for every column in cols_to_create_targets.
        """
        pass

class StockDataSet(DataSet):

    """
    Class to encapsuate a dataset. The dataset is essentially a list of dataframes of the
    same shape and size with the same features.

    In the case of stocks this is 1 or more stock dataframes that are combined into one dataset
    """

    def __init__(self, tickers, start, end=None, interval="1d"):
        super().__init__()
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
        self.feature_sets = [] # list of feature sets

    def create_dataset(self):
        """
        Create the dataset from the tickers and start/end dates and interval
        """
        if self.created_dataset:
            return

        self.dfs = []
        self.X_cols = set()
        self.y_cols = set()

        for ticker in self.tickers:
            stock_df, cols = get_stock_data(ticker, self.start, self.end, self.interval)
            self.dfs.append(stock_df)

        self.X_cols.update(cols)
        self.update_total_df()

        self.created_dataset = True

    def create_features(self):
        """
        Create additonal featues for the dataset
        """
        if self.created_features:
            return

        # Create price features
        for i in range(len(self.dfs)):
            df, feature_set = create_price_vars(self.dfs[i])
            self.dfs[i] = df
            self.feature_sets.append(feature_set)
            self.X_cols.update(feature_set.cols)

        # Create trend features
        for i in range(len(self.dfs)):
            df, feature_set = create_trend_vars(self.dfs[i])
            self.dfs[i] = df
            self.feature_sets.append(feature_set)
            self.X_cols.update(feature_set.cols)

        # Create percent change variables 
        for i in range(len(self.dfs)):
            df, feature_set = create_pctChg_vars(self.dfs[i])
            self.dfs[i] = df
            self.feature_sets.append(feature_set)
            self.X_cols.update(feature_set.cols)

        self.update_total_df()
        self.created_features = True

    def create_y_targets(self, cols_to_create_targets):
        """
        Create target y values for every column in cols_to_create_targets.

        NOTE - This is a specific implementation for stock data. The target values are gathered
        in a special way read @create_forward_rolling_sums for more info.
        """
        if self.created_y_targets:
            return

        for i in range(len(self.dfs)):
            df, cols = add_forward_rolling_sums(self.dfs[i], cols_to_create_targets)
            self.dfs[i] = df

        self.y_cols.update(cols)
        self.update_total_df()
        self.created_y_targets = True


def get_stock_data(
    ticker: str, start_date: datetime, end_date: datetime, interval="1d"
) -> pd.DataFrame:
    """
    Use the yfinance package and read the requested ticker from start_date to end_date. The following additional
    variables are created:

    additional variables: binarized weekday, month, binarized q1-q4

    All scaled variables are

    :param ticker:
    :param start_date:
    :param end_date:
    :param interval: 1d, 1wk, 1mo etc consistent with yfinance api
    :return:
    """

    df = pd.DataFrame()
    if end_date:
        df = yf.download([ticker], start=start_date, end=end_date, interval=interval)
    else:
        df = yf.download([ticker], start=start_date, interval=interval)

    df = df.drop(columns="Adj Close")
    # Standard column names needed for pandas-ta
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    return df, df.columns.tolist()

def create_price_vars(df: pd.DataFrame,moving_average_vals = [5,10,20,30,50,100],start_lag = 1, end_lag = 1) -> (pd.DataFrame, FeatureSet):
    """
    Create price features from the OHLC data.
    """

    feature_set = FeatureSet(ScalingMethod.SBSG)

    feature_set.cols +=["open", "high", "low", "close"]

    # Create price features
    for length in moving_average_vals:
        df["sma" + str(length)] = ta.sma(df.close, length=length)
        df["ema" + str(length)] = ta.ema(df.close, length=length)
        feature_set.cols.append("sma" + str(length))
        feature_set.cols.append("ema" + str(length))
        # df["sma" + str(length)].fillna(method='bfill', inplace=True)
        # df["ema" + str(length)].fillna(method='bfill', inplace=True)

    lag_df = create_lag_vars(df, feature_set.cols, start_lag, end_lag)
    df = pd.concat([df, lag_df], axis=1)
    feature_set.cols += lag_df.columns.tolist()
    
    for col in feature_set.cols:
        df[col] = df[col].fillna(method='bfill')

    return df, feature_set

def create_trend_vars(df: pd.DataFrame,start_lag = 1, end_lag = 1) -> (pd.DataFrame, FeatureSet):
    """
    Create trend features which are made up of features that are not price features but are not percent change features.
    """
    feature_set = FeatureSet(ScalingMethod.SBS)
    feature_set.cols += ["volume"]

    lag_df = create_lag_vars(df, feature_set.cols, start_lag, end_lag)
    df =  pd.concat([df, lag_df], axis=1)
    feature_set.cols += lag_df.columns.tolist()

    for col in feature_set.cols:
        # print(col)
        df[col] = df[col].fillna(df[col].mean())

    return df, feature_set
       

def create_pctChg_vars(
    df: pd.DataFrame, rolling_sum_windows=(1, 2, 3, 4, 5, 6), scaling_method = ScalingMethod.QUANT_MINMAX, start_lag = 1, end_lag = 1
) -> (pd.DataFrame, FeatureSet):
    """
    Create key target variables from the OHLC processed data.

    New variables created:

    The following variables are created based on the OHLC data:
    opHi, opLo, opCl, loCl, hiLo, hiCl.

    The following variables are created based on volume:
    minMaxVol, stdVolume, volChange (change from yesterday to today),

    pctChgClOp - percent change from yesterday's close to today's open
    pctChgClLo - percent change from yesterday's close to today's low
    pctChgClHi - percent change from yesterday's close to today's high
    pctChgClCl - percent change from yesterday's close to today's close

    :param df: A data frame containing OHLC data
    :param rolling_sum_windows: A list of summed window sizes for pctChgClCl to create, or None if not creating
    summed windows
    """
    feature_set = FeatureSet(scaling_method)

    for column in df.columns:
        df['pctChg' + column] = df[column].pct_change() * 100.0
        feature_set.cols.append('pctChg' + column)
    
    # % jump from open to high
    df['opHi'] = (df.high - df.open) / df.open * 100.0
    feature_set.cols.append('opHi')

    # % drop from open to low
    df['opLo'] = (df.low - df.open) / df.open * 100.0
    feature_set.cols.append('opLo')

    # % drop from high to close
    df['hiCl'] = (df.close - df.high) / df.high * 100.0
    feature_set.cols.append('hiCl')

    # % raise from low to close
    df['loCl'] = (df.close - df.low) / df.low * 100.0
    feature_set.cols.append('loCl')

    # % spread from low to high
    df['hiLo'] = (df.high - df.low) / df.low * 100.0
    feature_set.cols.append('hiLo')

    # % spread from open to close
    df['opCl'] = (df.close - df.open) / df.open * 100.0
    feature_set.cols.append('opCl')

    # Calculations for the percentage changes
    df["pctChgClOp"] = np.insert(np.divide(df.open.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClOp')

    df["pctChgClLo"] = np.insert(np.divide(df.low.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClLo')

    df["pctChgClHi"] = np.insert(np.divide(df.high.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClHi')

    if rolling_sum_windows:
        for roll in rolling_sum_windows:
            col_name = "sumPctChgclose_" + str(roll)
     
            df[col_name] = df.pctChgclose.rolling(roll).sum()
            feature_set.cols.append(col_name)

    lag_df = create_lag_vars(df, feature_set.cols, start_lag, end_lag)
    feature_set.cols += lag_df.columns.tolist()
    df = pd.concat([df, lag_df], axis=1)
    

    for col in feature_set.cols:
        # print(col)
        df[col] = df[col].fillna(df[col].mean())


    return df, feature_set

def create_weekday_month_cols(df):
    """
    Create binarized weekday and and continuous month columns
    """

    encoder = LabelBinarizer()
    wds = pd.DataFrame(
        data=encoder.fit_transform(df.index.weekday),
        index=df.index,
        columns=["Mon", "Tue", "Wed", "Thu", "Fri"],
    )
    cols_to_add = [col for col in wds.columns if col not in df.columns]

    # Concatenate only the unique columns from df_lags to df
    df = pd.concat([df, wds[cols_to_add]], axis=1)
    del wds

    df["month"] = pd.Series(data=df.index.month, index=df.index)

    return df


def create_quarter_cols(df):
    """
    Create a binarized quarter vector. For shorter datasets, some quarters may
    be missing, thus the extra check here
    """
    df_quarters = pd.get_dummies(df.index.quarter, prefix="q", prefix_sep="")
    df_quarters.index = df.index
    for q_missing in filter(
        lambda x: x not in df_quarters, ["q" + str(q) for q in range(1, 5)]
    ):
        df_quarters.loc[:, q_missing] = 0
    cols_to_add = [col for col in df_quarters.columns if col not in df.columns]

    # Concatenate only the unique columns from df_lags to df
    df = pd.concat([df, df_quarters[cols_to_add]], axis=1)
    del df_quarters

    return df


def add_forward_rolling_sums(df: pd.DataFrame, columns: list, scaling_method = ScalingMethod.QUANT_MINMAX, start_lag = 1, end_lag = 1) -> (pd.DataFrame, list):
    """
    Add the y val for sumPctChgClCl_X for the next X periods. For example sumPctChgClCl_1
    is the percent change from today's close to tomorrow's close, sumPctChgClCl_2 is the percent
    change from today's close to the close 2 days from now, etc. If we want to predict returns
    two days from now the new column sumPctChgClCl+2 would be the training target value.

    :param df: DataFrame to use
    :param columns: a list of column names
    :return: the DataFrame with the new columns added
    """
    new_columns = []
    for col in columns:
        # Extract the number X from the column name
        num_rows_ahead = int(col.split("_")[-1])

        # Create a new column name based on X
        new_col_name = "sumPctChgClCl+" + str(num_rows_ahead)
        new_columns.append(new_col_name)
        # Shift the column values by -X to fetch the value X rows ahead
        df[new_col_name] = df[col].shift(-num_rows_ahead)

    return df, new_columns


def create_lag_vars(
    df: pd.DataFrame, cols_to_create_lags: list, start_lag, end_lag
) -> list:
    """
    Create a DataFrame of lag variables

    :param df: DataFrame to use
    :param cols_to_create_lags: a list of column names to create lags for
    :param start_lag: start lag (default = 1)
    :param end_lag: end lag (inclusive, default = 1)
    :return: a list of the new lag variable column names
    """

    new_cols = {}

    for lag in range(start_lag, end_lag + 1):
        for var in cols_to_create_lags:
            if lag >= 1:
                col = var + "-" + str(lag)
                new_cols[col] = df[var].shift(lag)
            elif lag <= -1:
                col = var + "+" + str(-lag)
                new_cols[col] = df[var].shift(lag)

    # Convert new columns to a new DataFrame
    new_df = pd.DataFrame(new_cols)

    return new_df

