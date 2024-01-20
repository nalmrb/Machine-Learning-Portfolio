# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:47:31 2023

@author: nalmrb
"""

#for every model, load dataset consisting of one data point and time inference

from os.path import join
import define_paths

def load_data_single_trial(model_str, dataFilePath, chan = list(range(19))):
    
    if model_str == 'CSNN' or model_str == 'CNN':
        from tools import EEGDataset_CNN_CSNN
        dataset = EEGDataset_CNN_CSNN(dataFilePath, model_name = model_str,
                                      num_data_points = 10, chanInd = chan)
        
    elif model_str == 'GCN' or model_str == 'GCS' or model_str == 'GIN':
        from tools import EEGDataset_GNN
        dataset = EEGDataset_GNN(dataFilePath, num_data_points = 10,
                                 chanInd = chan)
        
    else:
        from tools import Load_exp1_data_EEGNet
        dataset, dataset_length = Load_exp1_data_EEGNet(dataFilePath, 
                                                        one_hot = True,
                                                        num_data_points = 10,
                                                        chanInd = chan)
    return dataset

def get_params(model_str):
    
    if model_str == 'CSNN':
        from snntorch import surrogate
        
        # Temporal Dynamics
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta = 0.5
        num_steps = 25
        
        params = {'spike_grad':spike_grad, 'beta':beta, 'num_steps':num_steps,
                  'model_name':model_str}
        
    elif model_str == 'CNN':
        params = {'model_name':model_str}
        
    elif model_str == 'GCS' or model_str == 'GCN' or model_str == 'GIN':
        params = {'model_name':model_str}
        
    else:
        params = {'model_name':model_str}
        
    return params

def load_model(params, dataset):
    from torch import nn as nn
    import snntorch as snn
    from torch import device
    from torch.cuda import is_available
    
    model_name = params.get('model_name')
    
    if model_name == 'CSNN':
        myDevice = device("cuda") if is_available() else device("cpu")
        
        #define params
        spike_grad = params.get('spike_grad')
        beta = params.get('beta')
        
        #load model
        #initialize model
        net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                        nn.Flatten(),
                        nn.Linear(15744, 2),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                        ).to(myDevice)
        
    elif model_name == 'CNN':
        myDevice = device("cuda") if is_available() else device("cpu")
        
        net = nn.Sequential(nn.Conv2d(1, 12, 5),
                        nn.MaxPool2d(2),
                        nn.Conv2d(12, 64, 5),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(15744, 2),
                        nn.Sigmoid()
                        ).to(myDevice)
        
    elif model_name == 'GCS':
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
        from spektral.layers import GlobalAttnSumPool, GCSConv
        
        #build model
        channels = dataset.n_node_features
        num_labels = dataset.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GCSConv(channels = channels//8)
                self.conv2 = GCSConv(channels = channels//16)
                self.conv3 = GCSConv(channels = channels//64)
                self.conv4 = GCSConv(channels = channels//128)
                self.conv5 = GCSConv(channels = channels//256)
                self.conv6 = GCSConv(channels = channels//512)
                self.pool = GlobalAttnSumPool()
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
        
        net = Net(channels, num_labels)
        
    elif model_name == 'GCN':
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
        from spektral.layers import GlobalAttnSumPool, GCNConv
        
        #build model
        channels = dataset.n_node_features
        num_labels = dataset.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GCNConv(channels = channels//8)
                self.conv2 = GCNConv(channels = channels//16)
                self.conv3 = GCNConv(channels = channels//64)
                self.conv4 = GCNConv(channels = channels//128)
                self.conv5 = GCNConv(channels = channels//256)
                self.conv6 = GCNConv(channels = channels//512)
                self.pool = GlobalAttnSumPool()
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
            
        net = Net(channels, num_labels)
        
    elif model_name == 'GIN':
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
        from spektral.layers import GlobalAttnSumPool, GINConv
        
        #build model
        channels = dataset.n_node_features
        num_labels = dataset.n_labels
        class Net(Model):
            def __init__(self, channels, num_labels):
                super().__init__()
                self.conv1 = GINConv(channels=channels//2, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv2 = GINConv(channels=channels//4, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv3 = GINConv(channels = channels//8, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv4 = GINConv(channels = channels//16, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv5 = GINConv(channels = channels//32, mlp_hidden = [256, 256, 256, 256, 256])
                self.conv6 = GINConv(channels = channels//64, mlp_hidden = [256, 256, 256, 256, 256])
                self.pool = GlobalAttnSumPool()
                # self.drop = Dropout(0.5)
                self.dense1 = Dense(512, activation = 'relu')
                self.dense2 = Dense(num_labels, activation="sigmoid")
                
            def call(self,inputs):
                x, a = inputs
                x = self.conv1([x,a])
                x = self.conv2([x,a])
                x = self.conv3([x,a])
                x = self.conv4([x,a])
                x = self.conv5([x,a])
                x = self.conv6([x,a])
                out = self.pool(x)
                # out = self.drop(out)
                out = self.dense1(out)
                out = self.dense2(out)
                
                return out
            
        #Create model
        net = Net(channels, num_labels)
        
    else:
        from EEGModels import EEGNet
        
        #configure model with defaults from example
        net = EEGNet(nb_classes = 2, Chans = 19, Samples = 996)
        
    return net


def get_inf_time(dataset, model, params):
    
    model_name = params.get('model_name')
    
    if model_name == 'CSNN':
        from torch.utils.data import DataLoader
        from torch import device
        from torch.cuda import is_available
        from tools import forward_pass
        from time import perf_counter
        
        myDevice = device("cuda") if is_available() else device("cpu")
        num_steps = params.get('num_steps')
        
        loader = DataLoader(dataset, batch_size = 1)
        
        train_batch= iter(loader)
        
        for data, targets in train_batch:
            data = data.to(myDevice)
            targets = targets.to(myDevice)
            
            model.train()
            time_start = perf_counter()
            spk_rec, mem_rec = forward_pass(model, num_steps, data)
            time_end = perf_counter()
            break
            
        inf_time = time_end - time_start
        
    elif model_name == 'CNN':
        from torch.utils.data import DataLoader
        from torch import device
        from torch.cuda import is_available
        from tools import forward_pass
        from time import perf_counter
        
        myDevice = device("cuda") if is_available() else device("cpu")
        
        loader = DataLoader(dataset, batch_size = 1)
        
        train_batch= iter(loader)
        
        for data, targets in train_batch:
            data = data.to(myDevice)
            targets = targets.to(myDevice)
            
            # forward pass
            model.train()
            time_start = perf_counter()
            out = model(data)
            time_end = perf_counter()
            break
            
        inf_time = time_end - time_start
        
    elif model_name == 'GCS' or model_name == "GCN" or model_name == "GIN":
        from spektral.data.loaders import MixedLoader
        from time import perf_counter
        
        if model_name == "GCN" or model_name == "GCS":
            from spektral.transforms import  GCNFilter
            dataset.apply(GCNFilter())
        elif model_name == "GIN":
            from spektral.layers import GINConv
            from spektral.transforms import  LayerPreprocess
            from spektral.utils.sparse import sp_matrix_to_sp_tensor
            dataset.apply(LayerPreprocess(GINConv))
            dataset.a = sp_matrix_to_sp_tensor(dataset.a)
            
        loader = MixedLoader(dataset, batch_size = 1)
        
        for batch in loader:
            inputs, targets = batch
            time_start = perf_counter()
            predictions = model(inputs)
            time_end = perf_counter()
            break
            
        inf_time = time_end - time_start
        
    else:
        from time import perf_counter
        dataset = dataset.batch(1)
        
        for step, (inputs, targets) in enumerate(dataset):
            time_start = perf_counter()
            predictions = model(inputs)
            time_end = perf_counter()
            break
            
        inf_time = time_end - time_start
        
    return inf_time  
        

#define model strings
models = ['CSNN', 'CNN', 'EEGNet', 'GCS', 'GCN', 'GIN']

#define paths
paths = define_paths.main()

chan = [2,3,10,13,14];

#get inference times
inf_times = []
for model_str in models:
    
    dataset = load_data_single_trial(model_str, paths.get('dataFilePath'), 
                                     chan = chan)
    
    params = get_params(model_str)
    
    model = load_model(params, dataset)
    
    inf_time = get_inf_time(dataset, model, params)
    
    inf_times.append(inf_time)
    
#write to file
res_file = open(join(paths.get('results_path'), 'inference_times.txt'), 'w')
res_file.write('Inference times per model\n')
res_file.write('\n')
cnt = -1
for model_str in models:
    cnt += 1
    res_file.write(f'{model_str}: ')
    res_file.write(f'{inf_times[cnt]}\n')
    res_file.write('\n')
    
res_file.close()
    