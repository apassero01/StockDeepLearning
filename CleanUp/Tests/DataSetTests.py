import sys
sys.path.append('../')

import TSeriesPreproccesing as TSPP
import unittest

class TestStockDataSet(unittest.TestCase):

    def setUp(self):
        '''
        This method is called before each test
        '''
        self.tickers = ['AAPL', 'MSFT', 'AMD']
        self.start_date = '2010-01-01'
        self.stockDataSet = TSPP.StockDataSet(self.tickers, self.start_date)

    ## Non Class Methods
    def test_create_price_vars(self):
        '''
        Test that the non class function create_price_vars works as expected 
        This function is outside of the class and takes a dataframe as an input and moving averages values with default values of 5, 10, 20, 50, 100]

        Tests the following: 
            - The dataset contains the following columns: 
                - 'open', 'high', 'low', 'close' sma5, sma10, ... ema5, ema10....
                - There are no nan values in the dataset 
                - A non empty feature set is returned and the cols in the feature set are all in the dataframe

        '''
        df = self.stockDataSet.dfs[0]
        df, feature_set = TSPP.create_price_vars(df) # Calling the create_price_vars function that takes a df and returns the updated df with the new feature set
    
    def test_create_trend_vars(self):
        '''
        Test that the non class function create_trend_vars works as expected 
        This function is outside of the class and takes a dataframe as an input 

        Tests the following: 
            - The dataset contains the volume as a new column 
            - There are no nan values in the dataset 
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe

        '''
        df = self.stockDataSet.dfs[0]
        df, feature_set = TSPP.create_trend_vars(df)
    
    def test_create_pctChg_vars(self):
        '''
        Test that the non class function create_pctChg_vars works as expected
        This function is outside of the class and takes a dataframe as an input

        Tests the following: 
            - The dataset contains new columns for all new percent change variables (don't have to test for each specific column but make sure that the columns are there) 
            - Test a few randomly selected columns ie. pctChgclose etc and make sure that the values are correct by extracting the actual values from the dataframe
                - take a random day and calculate the pctChgclose manually and compare it to the value in the dataframe. close on 5/10/21 - close on 5/9/21 / close on 5/9/21 * 100 == pctChgclose on 5/10/21
                - Do this for a few random days and columns including the single day pct chg vars such as opHi (pct change from open to high) etc 
            - More Difficult: Confirm rolling sum variables are correct. This function will return sumPctChgclose_1 -> sumPctChgclose_6 which consists of the sum of pctChgClose from today to t-1, t-2, t-3, t-4, t-5, t-6 
            respectively. sumPctChgclose_1 is the same as pctChgClose and sumPctChgclose_6 is the sum of pctChgClose from today to 6 days ago.

            - There are no nan values in the dataset
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe
        '''
        df = self.stockDataSet.dfs[0]
        df, feature_set = TSPP.create_pctChg_vars(df)
    
    def test_add_forward_rolling_sim(self):
        '''
        Test that the non class function add_forward_rolling_sim works as expected
        This function is outside of the class and takes a dataframe as an input

        Tests the following: 
            - The dataset contains new columns for all new forward rolling sum variables
            - This function takes the sumPctChgClose_X variables and creates columns with the future sum of pctChgClose from t+1 to t+X days. (used as a target variable)
            - Difficult : Confirm these values are correct by manually calculating sumPctChgClose+X for a few random days and comparing them to the values in the dataframe. 
                - if today is t, sumPctChgClose+1 is sumPctChgClose_1 for t+1, sumPctChgClose+2 is sumPctChgClose_2 for t+2, etc. (diagnol of the dataframe)
            - There are no nan values in the dataset
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe
        '''
        df = self.stockDataSet.dfs[0]
        df, feature_set = TSPP.add_forward_rolling_sum(df)

    
    ## Class Methods
    def test_create_stock_dataset(self):
        '''
        Test that the stock dataset is created using stockDataSet.create_dataset() method 
        Tests the following: 
            - The dataset is not None
            - The length of stockDataSet.tickers is equal to the length of self.tickers
            - The length of stockDataSet.dfs = len(self.tickers)
            - The length of stockDataSet.dfs[i] = len(self.tickers[i]) for all i in range(len(self.tickers))
            - stockDataSet.cols = ['open', 'high', 'low', 'close', 'volume']
        '''
        self.assertIsNotNone(self.stockDataSet)
        self.stockDataSet.create_dataset()
    
    def test_create_features(self):
        '''
        Test that the feature set is created using stockDataSet.create_features() method. 
        This method calls all of the previously tested non class methods. Run tests to confirm it is done correctly

        Tests the following: 
            - The dataset is not None
            - This method calls create_price_vars, create_trend_vars, create_pctChg_vars functions ensure that the individual outputs from these functions are reflected in the dataset object after this method is called
                - ie. stockDataSet.dfs[0] contains all the new columns created by these functions
                - ie. stockDataSet.X_feature_sets contains all feature sets we expected to see earlier returned from non class functions
                - if we are running tests with multiple tickers, is df[0] the same as df[1] etc. (has all same columns, dates, num_rows, for each different stock)
            - X_cols contains all of the column names that we expected from individual functions
            - There are no nan values in the dataframe
        '''
        self.assertIsNotNone(self.stockDataSet)
        self.stockDataSet.create_dataset()

        # Pull out one of the data frames as a copy to test passing into non class functions is the same as calling create_features method
        df_test = self.stockDataSet.dfs[0].copy()
        
        self.stockDataSet.create_features()

        # Test that the individual functions are called and the outputs are reflected in the dataset object
        df_test, price_feature_set = TSPP.create_price_vars(df_test)
        df_test, trend_feature_set = TSPP.create_trend_vars(df_test)
        df_test, pctChg_feature_set = TSPP.create_pctChg_vars(df_test)

        # Test that the feature sets are the same as the ones returned from the individual functions
        self.assertEqual(self.stockDataSet.X_feature_sets[0], price_feature_set + trend_feature_set + pctChg_feature_set)

        #TODO add a lot more tests like this to confirm everything works as expected
    
    def test_create_target(self):
        '''
        Test that the target set is created using stockDataSet.create_target() method. This method calls the add_forward_rolling_sum function
        tested before. Run similar tests to confirm it is done correctly and the dataframe contains the target columns we expect

        Tests the following: 
            - The dataset is not None
            - This method calls add_forward_rolling_sum function ensure that the individual outputs from this function are reflected in the dataset object after this method is called
                - ie. stockDataSet.dfs[0] contains all the new columns created by this function
                - ie. stockDataSet.y_feature_sets contains all feature sets we expected to see earlier returned from non class functions
                - if we are running tests with multiple tickers, is df[0] the same as df[1] etc. (has all same columns, dates, num_rows, for each different stock)
            - y_cols contains all of the column names that we expected from individual functions
            - There are no nan values in the dataframe
            - stockDataSet.y_cols contains the columns we expect. ie. sumPctChgClose+1, sumPctChgClose+2, etc.
        '''

        self.assertIsNotNone(self.stockDataSet)

        self.stockDataSet.create_dataset()

        # Pull out one of the data frames as a copy to test passing into non class functions is the same as calling create_features method
        df_test = self.stockDataSet.dfs[0].copy()
        df_test = TSPP.add_forward_rolling_sum(df_test)

        self.stockDataSet.create_features()

        # mentually generate the columns we will be using to generate target columns 
        target_cols = [col for col in self.stockDataSet.df.columns if 'sumPctChgclose_' in col and '-' not in col]

        self.stockDataSet.create_y_targets(target_cols)
        


    


