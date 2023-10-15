from datetime import datetime

class ClusterGroupParams: 
    def __init__(self, start_date,end_date = None): 
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")

        self.end_date = end_date
        if self.end_date is None:
            self.end_date = datetime.today().date()

        self.X_feature_dict = None 
        self.y_feature_dict = None
        self.X_cols = None
        self.y_cols = None
        self.X_feature_sets = None
        self.y_feature_sets = None
        self.train_seq_elements = None
        self.test_seq_elements = None

class StockClusterGroupParams(ClusterGroupParams): 

    def __init__(self, start_date, tickers, target_cols, n_steps, interval = '1d', end_date = None): 
        super().__init__(start_date, end_date)
        self.tickers = tickers
        self.target_cols = target_cols
        self.n_steps = n_steps
        self.interval = interval



class ClusterPeriod: 
    def __int__(self,start_date,end_date): 
        self.start_date = start_date
        self.end_date = end_date
    

class ClusterGroup: 
    def __init__(self): 
        pass



        
