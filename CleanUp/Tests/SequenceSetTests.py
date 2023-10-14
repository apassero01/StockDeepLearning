import unittest
import sys
sys.path.append('../')
import TSeriesPreproccesing as TSPP
import SequencePreprocessing as SP

class TestStockSequenceSet(unittest.TestCase):

    def setUp(self) -> None:
        '''
        This method is called before each test
        '''
        self.tickers = ['AAPL', 'MSFT', 'AMD']
        self.start_date = '2010-01-01'
        self.stockDataSet = TSPP.StockDataSet(self.tickers, self.start_date)
        
        #Create Dataset and do preprocessing 
        self.stockDataSet.create_dataset()
        self.stockDataSet.create_features()
        self.stockDataSet.create_y_targets()

        #Data set creates a sequence set
        self.stockSequenceSet = self.stockDataSet.create_sequence_set()
    
    def test_train_test_split(self):
        '''
        Tests that the train test split function works as expected

        Tests the following:
            - train and test sets are not empty 
            - The last date in the training set happens before the first day in the test set 
        '''
    
    def scale_quant_min_max(self):
        '''
        Tests that the scale_quant_min_max function works as expected
        Example tests for this are in SequenceScalingTESTS.ipynb

        Tests the following:
            - All features with featureSet.scalingMethod == QUANT_MINMAX are scaled between -1 and 1
            - For every feature, the number of instances > 0 and < 0 are the same before and after calling the method
            - The scaler is saved in feature_set.scaler and inverse transforming with the scaler returns original values (close to)
                - test this for a few random rows in the dataframe
        '''
        pass
    
    def test_create_sequence(self):
        '''
        Test that the non class function create_sequence works as expected

        Tests the following:
            - X_feature_dict contains key value pairs of the form {feature_name: feature_index}
                - To test this extract some data points from the dataframe and ensure that the same 
                value exists in the newly created sequenceElement in the index location indicated by the dict
                This Test must confirm that the feature names are correctly mapped to index location
            - y_target_dict contains key value pairs of the form {target_name: target_index} 
                - test to confirm that index locations of sumPctChgclose+1, +2... are in that order exactly in feature element
            - The sequence contains the correct value 
                - Test these by manually creating a few SequenceElements from the dataframe and compare the sequences to the returned sequences
        '''
        df = self.stockSequenceSet.dfs[0]
        X_cols = self.stockSequenceSet.X_cols
        y_cols = self.stockSequenceSet.y_cols
        n_steps = 10 # Whatever you would like 
        ticker = self.stockSequenceSet.tickers[0]
        sequence_elements, X_feature_dict, y_feature_dict = SP.create_sequence(df, X_cols, y_cols, n_steps, ticker)
        # TODO write tests above to ensure returned sequence elements are correct by comparing to values in the dataframe

    
    def test_create_comined_sequence(self):
        '''
        Test that the create_combined_sequence function works as expected

        Tests the following:
            - there are sequence_elements for every ticker in tickers list
            - the X_feature_dict and y_feature_dict contains feature name and index pairs for every feature in the sequence
            
        '''
        pass

    def test_scale_sbs(self):
        '''
        Test that the scale_sbs function works as expected
        Recall scaling sequence by sequence (SBS) means each sequenceElement is scaled independently of the others
        Example tests for this are in SequenceScalingTESTS.ipynb


        Tests the following:
            - All features with featureSet.scalingMethod == SBS are scaled between FeatureSet.range 
            - Ensure scaled data shape is the same as the original data shape
                - This can be done visually by looking at plotted points that should look the same but we want automated test
                - Should be able to do this by finding the correlation between the original and scaled data and that value should be -1 or 1 exactly. 
            - Each sequence element has a list of feature_sets and each feature set has a scaler saved to it. Extract the scaler with scaling_method = sbs and then  
            and inverse transform with the scaler and confirm it returns the original values in non scaled set (close to)
                - test this for a few random rows in the dataframe
        '''
        pass

    def test_scale_sbsg(self):
        '''
        Test that the scale_sbsg function works as expected
        Recall scaling sequence by sequence grouped (SBSG) means each sequenceElement is scaled independently of other sequences but features in the same sequence are scaled together
        For example if we have the 10day moving average and close price we want those features to be on the same scale but it is important that
        the distance between the two is preserved.

        Example tests for this are in SequenceScalingTESTS.ipynb

        Tests the following:
            - All features with featureSet.scalingMethod == SBSG are scaled between FeatureSet.range 
            - Ensure scaled data shape is the same as the original data shape for all features in feature sets 
                - This can be done visually by looking at plotted points that should look the same but we want automated test
                - Should be able to do this by finding the correlation between the original and scaled data and that value should be -1 or 1 exactly. 
                - might have to check correlation for each feature in feature set individually 
            - Each sequence element has a list of feature_sets and each feature set has a scaler saved to it. Extract the scaler with scaling_method = sbsg and then  
            and inverse transform with the scaler and confirm it returns the original values in non scaled set (close to)
                - test this for a few random rows in the dataframe
        '''
        pass

    def test_y_scale_sbs(self):
        '''
        Test that the y_scale_sbs function works as expected. For a 1D array of y vales in each sequence element, scale each sequence element independently of the others

        Tests the following: 
            - All y values in each sequence element are scaled between -1 and 1
            - Each sequenceElement.y_seq should contain a value of -1 and 1 exactly
            - Ensure scaled data shape is the same as the original data 
                - again use correlation to test this
            - Each sequence element has a scaler saved to it. Extract the scaler and then inverse transform to confirm
            we get the original values matching unscaled seq elements
                - test this for a few random rows in the dataframe
        '''

    def test_scale_and_sequence(self):
        '''
        Test that the scale_and_sequence function works as expected
        This function is a wrapper for calling sequence methods requiring dataframe form first, then creates sequence, then scales the sequence by sequence features
        The tests for the internal function calls are done above but we need a few additional tests to ensure the wrapper works as expected

        Tests the following: 
            - All values in each sequence element are scaled between -1 and 1
            - Each sequenceElement has a list of feature sets the same length as X_feature_sets
                - there should also be the same number of QUANT_MINMAX, SBS and SBSG feature sets in the scaler.X_feature_sets list 
            - The length of each sequenceElement.X_seq and sequenceElement.y_seq is the same as the length
        '''
        

    


    

        