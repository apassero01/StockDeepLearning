from datetime import datetime
from SequencePreprocessing import StockSequenceSet, SequenceElement
from TSeriesPreproccesing import StockDataSet
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
import plotly.graph_objects as go
import math
from keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU,BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd

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

    def __init__(self, start_date, tickers, target_cols, n_steps, cluster_features, interval = '1d', end_date = None): 
        super().__init__(start_date, end_date)
        self.tickers = tickers
        self.target_cols = target_cols
        self.n_steps = n_steps
        self.interval = interval
        self.cluster_features = cluster_features



class ClusterPeriod: 
    def __int__(self,start_date,end_date): 
        self.start_date = start_date
        self.end_date = end_date
        self.current_cluster_group = None
    
    def create_cluster_group(self,group_params): 
        pass

class StockClusterPeriod(ClusterPeriod):
    
    def create_cluster_group(self, group_params):
        pass


class ClusterGroup: 
    def __init__(self,group_params = None): 
        self.group_params = group_params
    
    def set_group_params(self,group_params): 
        self.group_params = group_params

    def create_data_set(self): 
        pass

    def create_sequence_set(self):
        pass

class StockClusterGroup(ClusterGroup):
    def __init__(self,group_params = None): 
        super().__init__(group_params=group_params)
        self.data_set = None
        self.sequence_set = None

    def create_data_set(self):
        self.data_set = StockDataSet(self.group_params)
        self.data_set.preprocess_pipeline() 
        self.group_params = self.data_set.group_params

    
    def create_sequence_set(self):
        self.sequence_set = self.data_set.create_sequence_set() 
        self.sequence_set.preprocess_pipeline(add_cuma_pctChg_features=True)
        self.group_params = self.sequence_set.group_params
    
    def run_clustering(self,alg = 'TSKM',metric = "euclidean"):
        
        self.get_3d_array()
        X_train_cluster = self.filter_by_features(self.X_train, self.group_params.cluster_features)
        X_test_cluster = self.filter_by_features(self.X_test, self.group_params.cluster_features)

        print(X_train_cluster.shape)
        print(X_test_cluster.shape)


        n_clusters = math.ceil(math.sqrt(len(X_train_cluster))) // 4

        if alg == 'TSKM':
            self.cluster_alg = TimeSeriesKMeans(n_clusters=n_clusters, metric=metric,random_state=3)
        
        self.train_labels = self.cluster_alg.fit_predict(X_train_cluster)
        self.test_labels = self.cluster_alg.predict(X_test_cluster)

        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        if len(self.train_labels) != len(train_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")
        if len(self.test_labels) != len(test_seq_elements):
            raise ValueError("The number of labels does not match the number of sequences")

        for i in range(len(train_seq_elements)):
            seq_element = train_seq_elements[i]
            seq_element.cluster_label = self.train_labels[i]
        for i in range(len(test_seq_elements)):
            seq_element = test_seq_elements[i]
            seq_element.cluster_label = self.test_labels[i]
    
    def create_clusters(self):
        train_seq_elements = self.group_params.train_seq_elements
        test_seq_elements = self.group_params.test_seq_elements

        train_labels = np.unique([x.cluster_label for x in train_seq_elements])

        self.clusters = [] 

        for label in train_labels:
            cur_train_seq_elements = [x for x in train_seq_elements if x.cluster_label == label]
            cur_test_seq_elements = [x for x in test_seq_elements if x.cluster_label == label]
            self.clusters.append(StockCluster(self,cur_train_seq_elements,cur_test_seq_elements))


    def display_all_clusters(self):
        for cluster in self.clusters:
            fig = cluster.visualize_cluster()
            fig.show()
    
    def train_all_rnns(self,model_features):
        for cluster in self.clusters:
            cluster.train_rnn(model_features)


    def filter_by_features(self,seq, feature_list):
        seq = seq.copy()
        indices = [self.group_params.X_feature_dict[x] for x in feature_list]
        # Using numpy's advanced indexing to select the required features
        return seq[:, :, indices]
    
    def get_3d_array(self): 
        self.X_train, self.y_train, self.X_test, self.y_test = self.sequence_set.get_3d_array()
        return self.X_train, self.y_train, self.X_test, self.y_test


class Cluster:
    def __init__(self, cluster_group,train_seq_elements,test_seq_elements):
        self.cluster_group = cluster_group
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
    
    def get_3d_array(self):
        self.X_train, self.y_train = SequenceElement.create_array(self.train_seq_elements)
        self.X_test, self.y_test = SequenceElement.create_array(self.test_seq_elements)
        return self.X_train, self.y_train, self.X_test, self.y_test

class StockCluster(Cluster):
    def __init__(self,cluster_group, train_seq_elements, test_seq_elements):
        super().__init__(cluster_group,train_seq_elements,test_seq_elements)
        self.cluster_group = cluster_group
        self.train_seq_elements = train_seq_elements
        self.test_seq_elements = test_seq_elements
        self.label = train_seq_elements[0].cluster_label
        self.get_3d_array()
    
    def remove_outliers(self):
        pass

    def visualize_cluster(self, isTrain = True):
        if isTrain:
            arr_3d = self.X_train
        else:
            arr_3d = self.X_test

        X_cluster = self.cluster_group.filter_by_features(arr_3d, self.cluster_group.group_params.cluster_features)

        traces = [] 
        avg_cluster = np.mean(X_cluster,axis = 0)
        std_cluster = np.std(X_cluster,axis = 0)

        upper_bound = avg_cluster + std_cluster
        lower_bound = avg_cluster - std_cluster
        
        x = np.arange(avg_cluster.shape[0])

        for feature_idx in range(avg_cluster.shape[1]):
            y_avg = np.ones(avg_cluster.shape[0]) * feature_idx
            z_avg = avg_cluster[:, feature_idx]
            z_upper = upper_bound[:, feature_idx]
            z_lower = lower_bound[:, feature_idx]
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_avg, mode='lines', line=dict(color='red', width=2)))
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_upper, mode='lines', line=dict(color='green', width=1)))
            traces.append(go.Scatter3d(x=x, y=y_avg, z=z_lower, mode='lines', line=dict(color='blue', width=1)))

        fig = go.Figure(data=traces)

        fig.update_layout(title="Cluster " + str(self.label),
                        scene=dict(xaxis_title='Time',
                                    yaxis_title='Feature Index',
                                    zaxis_title='Value'))

        return fig
    
    def train_rnn(self,model_features):
        if len(self.X_train) == 0 or len(self.X_test) == 0:
            return
        X_train_filtered = self.cluster_group.filter_by_features(self.X_train, model_features)
        X_test_filtered = self.cluster_group.filter_by_features(self.X_test, model_features)
        y_train = self.y_train
        y_test = self.y_test


        self.model = create_model(len(model_features))
        
        self.model.fit(X_train_filtered, y_train, epochs=250, batch_size=32, validation_data=(X_test_filtered, y_test), verbose=1)

        predicted_y = self.model.predict(X_test_filtered)
        predicted_y = np.squeeze(predicted_y, axis=-1)

        results = pd.DataFrame({
            'predicted_1': predicted_y[:, 0],
            'predicted_2': predicted_y[:, 1],
            'predicted_3': predicted_y[:, 2],
            'predicted_4': predicted_y[:, 3],
            'predicted_5': predicted_y[:, 4],
            'predicted_6': predicted_y[:, 5],
            'real_1': y_test[:, 0],
            'real_2': y_test[:, 1],
            'real_3': y_test[:, 2],
            'real_4': y_test[:, 3],
            'real_5': y_test[:, 4],
            'real_6': y_test[:, 5],
        })

        results['same_1'] = ((results['predicted_1'] > 0) & (results['real_1'] > 0)) | ((results['predicted_1'] < 0) & (results['real_1'] < 0))
        results['same_2'] = ((results['predicted_2'] > 0) & (results['real_2'] > 0)) | ((results['predicted_2'] < 0) & (results['real_2'] < 0))
        results['same_3'] = ((results['predicted_3'] > 0) & (results['real_3'] > 0)) | ((results['predicted_3'] < 0) & (results['real_3'] < 0))
        results['same_4'] = ((results['predicted_4'] > 0) & (results['real_4'] > 0)) | ((results['predicted_4'] < 0) & (results['real_4'] < 0))
        results['same_5'] = ((results['predicted_5'] > 0) & (results['real_5'] > 0)) | ((results['predicted_5'] < 0) & (results['real_5'] < 0))
        results['same_6'] = ((results['predicted_6'] > 0) & (results['real_6'] > 0)) | ((results['predicted_6'] < 0) & (results['real_6'] < 0))
        accuracy1 = results['same_1'].sum() / len(results) * 100
        accuracy2 = results['same_2'].sum() / len(results) * 100
        accuracy3 = results['same_3'].sum() / len(results) * 100
        accuracy4 = results['same_4'].sum() / len(results) * 100
        accuracy5 = results['same_5'].sum() / len(results) * 100
        accuracy6 = results['same_6'].sum() / len(results) * 100

        output_string = (
        "Cluster Number: " + str(self.label) +
        " \nAccuracy1D " + str(accuracy1) + " PredictedRet: " + str(results['predicted_1'].mean()) + " ActRet " + str(results['real_1'].mean() ) +
        " \nAccuracy2D " + str(accuracy2) + " PredictedRet: " + str(results['predicted_2'].mean()) + " ActRet " + str(results['real_2'].mean() ) +
        " \nAccuracy3D " + str(accuracy3) + " PredictedRet: " + str(results['predicted_3'].mean()) + " ActRet " + str(results['real_3'].mean() ) +
        " \nAccuracy4D " + str(accuracy4) + " PredictedRet: " + str(results['predicted_4'].mean()) + " ActRet " + str(results['real_4'].mean() ) +
        " \nAccuracy5D " + str(accuracy5) + " PredictedRet: " + str(results['predicted_5'].mean()) + " ActRet " + str(results['real_5'].mean() ) +
        " \nAccuracy6D " + str(accuracy6) + " PredictedRet: " + str(results['predicted_6'].mean()) + " ActRet " + str(results['real_6'].mean() ) +
        " Train set length: " + str(len(X_train_filtered))+ " Test set length: " + str(len(y_test)) + "\n"
        )

        with open('output.txt', 'a') as f:
            f.write(output_string)









def create_model(input_shape):
    # Encoder
    model_lstm = Sequential()
    
    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True, input_shape=(None, input_shape)))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=100, activation='tanh',return_sequences=True))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=100, activation='tanh'))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))
    
    # Repeat the encoder output (which is the last hidden state) 
    # for 'n' times where 'n' is the number of prediction steps
    model_lstm.add(RepeatVector(6))  # Assuming you are predicting for 6 steps

    # Decoder
    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))

    model_lstm.add(LSTM(units=250, activation='tanh', return_sequences=True))
    model_lstm.add(BatchNormalization())
    model_lstm.add(Dropout(0.2))
    
    model_lstm.add(TimeDistributed(Dense(1)))  # Predict one value for each time step

    optimizer = Adam(learning_rate=0.001)

    model_lstm.compile(optimizer=optimizer, loss="mse")
    return model_lstm

    



        
