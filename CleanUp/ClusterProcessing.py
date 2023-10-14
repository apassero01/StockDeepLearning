
class ClusterGroupParams: 
    def __init__(self, start_date): 
        self.start_date = start_date

class StockClusterGroupParams(ClusterGroupParams): 
    def __init__(self, start_date, tickers, target_cols, interval = '1d'): 
        super().__init__(start_date)
        self.tickers = tickers
        self.target_cols = target_cols
        self.interval = interval


class ClusterPeriod: 
    def __int__(self,start_date,end_date): 
        self.start_date = start_date
        self.end_date = end_date
    

class ClusterGroup: 
    def __init__(self): 



        
