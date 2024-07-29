# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:52:42 2023

@author: nalmrb
"""

def load_data(paths, params):
    
    from tensorflow.data import Dataset
    import os
    from glob import glob
    import numpy as np
    from sklearn.preprocessing import minmax_scale
    from re import findall
    
    chanInds = params.get('chanInds', list(range(19)))
    #options based on stage
    stage = params.get('stage')
    if stage == 'few_shot_learning':
        datasets = []
        dataset_lengths = []
        from tools_exp3 import augment_data
        num_iter = 5
    else:
        num_iter = 1
    test_set_stages = ['get_inf_times','few_shot_learning']
    if stage in test_set_stages:
        dir_idx_start = params.get('num_train_drivers')
        dir_idx_end   = 11
    else:
        dir_idx_start = 0
        dir_idx_end   = params.get('num_train_drivers')
        
    #define different permuts
    driver_permuts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      [4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3],
                      [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6],
                      [10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1],
                      [5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4],
                      [8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7],
                      [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2],
                      [6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5]]
    
    #construct file list
    load_folder_nums = driver_permuts[params.get('bootstrap_idx')][dir_idx_start:dir_idx_end]
    filePathList = []
    folders = os.listdir(paths.get('data_path'))
    for this_dir in folders:
        if os.path.isdir(os.path.join(paths.get('data_path'), this_dir)):
            #get number
            driver_num = findall(r'\d+',this_dir)
            if eval(driver_num[0]) in load_folder_nums:
                filePathList.append(os.path.join(paths.get('data_path'), this_dir))         
    
    #create empty lists for storage
    data_list  = []
    label_list = []
    
    print(filePathList)
    for this_filePath in filePathList:
        
        #get directory contents
        fileList = glob(os.path.join(this_filePath, '*.npy'))
        
        #loop through data
        for data_file_idx in range(len(fileList)):
            for data_iter in range(num_iter):
                #read data
                data = np.load(fileList[data_file_idx])
                #drop channels not used
                if len(chanInds)<19:
                    chanToDel = [x for x in list(range(19)) if x not in chanInds]
                    data = np.delete(data, chanToDel, axis = 1)
                #manipulate data
                data = np.transpose(data)
                if data_iter > 0:
                    data = augment_data(data)
                data = minmax_scale(data, axis=1)
                #convert to 8bit integer
                data = data * 255
                data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1],1])
                data = data.astype(np.uint8)
                #add back data as zeros if channels were deleted
                for chan_idx in range(19):
                    if chan_idx not in chanInds:
                        data = np.pad(data, ((0,1),(0,0),(0,0)))
                
                #save data to list
                data_list.append(data)
                
                #determine file label
                if "NoGo" in fileList[data_file_idx]:
                    if params.get('one_hot'):
                        label = np.array([1,0])
                    else:
                        label = np.int32(0)
                else:
                    if params.get('one_hot'):
                        label = np.array([0,1])
                    else:
                        label = np.int32(1)
                        
                #save label to list
                label_list.append(label)
            
            #stop short if debug
            if params.get('debug'):
                if len(data_list) >= 50:
                    break
                
        if stage == 'few_shot_learning':
            
            # change into tensorflow dataset
            data_array = np.array(data_list)
            label_array = np.array(label_list)
            dataset = Dataset.from_tensor_slices((data_array, label_array))
            
            datasets.append(dataset)
            dataset_lengths.append(len(data_list))
            
            #define storage
            data_list = []
            label_list = []
                
        #stop short if debug
        if params.get('debug'):
            if len(data_list) >= 50:
                break
            
    if stage != 'few_shot_learning':
        # change into tensorflow dataset
        data_array = np.array(data_list)
        label_array = np.array(label_list)
        dataset = Dataset.from_tensor_slices((data_array, label_array))
    
        return dataset, len(data_list)
    else:
        return datasets, dataset_lengths
    
def augment_data(data):
    import numpy as np
            
    #get shape of data sample
    data_shape = np.shape(data)
    
    #create gaussian noise
    noise = np.random.normal(0, 1, data_shape)
    
    #add to data
    aug_data = data + noise
    
    return aug_data

def supervised_learning(paths, params, net = None):
    
    import tensorflow as tf
    from models_exp3 import build_and_compile_CNN
    from tools_exp3 import load_data
    from os.path import join
    import keras
    
    #define dataset
    dataset, dataset_length = load_data(paths, params)
    
    #batch and shuffle dataset
    if params.get('batch_size') == -1:
        batch_size = dataset_length
    else:
        batch_size = params.get('batch_size')
    dataset = dataset.shuffle(params.get('shuffle_buffer_size')).batch(batch_size)
    
    #build and compile network
    if net is None:
        if params.get('network') == 'CNN':
            dummy_input_shape = [1,19,1074,1]
            net, custom_metrics = build_and_compile_CNN(dummy_input_shape)
            
    #define custom callback for stopping model training if 90% metric score is
    #achieved for all metric
    class check_metric_scores_early_stop(tf.keras.callbacks.Callback):
        def __init__(self,metric_targ):
            super().__init__()
            self.metric_targ = metric_targ 
            
        def on_epoch_end(self, epoch, logs=None):
            #get all metric performance
            acc = logs.get("accuracy")
            tpr = logs.get('TPR')
            tnr = logs.get('TNR')
            
            #all above metric target, stop training
            if acc >= self.metric_targ and tpr >= self.metric_targ and tnr >= self.metric_targ:
                self.model.stop_training = True
        
    #fit model
    data_point = next(iter(dataset))
    if params.get('batch_size') != tf.shape(data_point[0])[0]:
        batch_size = tf.shape(data_point[0])[0]
    else:
        batch_size = params.get('batch_size')
        
    #setup checkpoint
    chkpt_path = join(paths.get('models_path'),'cp.ckpt')
    model_chkpt_callback = keras.callbacks.ModelCheckpoint(
        filepath = chkpt_path,
        monitor = 'loss',
        mode = 'min',
        save_best_only = True,
        save_weights_only = True)
    
    #define early stop based on loss callback (to save time)
    loss_early_stop_callback = keras.callbacks.EarlyStopping(
        monitor = "loss",
        min_delta = 1e-3,
        patience = 10,
        restore_best_weights = True,
        start_from_epoch = 30)
    
    #define custom callback object if quantize retrain
    if params.get('stage') == "quantize":
        check_metric_scores_early_stop_obj = check_metric_scores_early_stop(metric_targ = params.get('metric_target'))
        call_backs = [check_metric_scores_early_stop_obj]
    elif params.get('stage') == 'train':
        call_backs = [model_chkpt_callback]
    
    train_hist = net.fit(dataset, epochs = params.get('max_epochs'), 
                         batch_size = batch_size,
                         callbacks = call_backs)
    # if params.get('stage') == 'train':
    #     #load back best weights
    #     net.load_weights(chkpt_path)
    
    #eval network on dataset
    eval_score = net.evaluate(dataset)
    
    #save model
    from os.path import join
    if params.get('stage') == 'train':
        save_name = 'exp3_init_trained_network.keras'
    elif params.get('stage'):
        save_name = 'exp3_quantized_network.keras'
        
    net.save(join(paths.get('models_path'), save_name))
    
    return net, train_hist, eval_score
        
    
def plot_metrics_from_hist(history, save_path):
    
    import matplotlib.pyplot as plt
    from os.path import join
    
    #plot loss
    plt.plot(history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.savefig(join(save_path, "loss.png"))
    plt.close()
    
    #create subplots
    fig, axs = plt.subplots(3)
    for axs_ind in range(len(axs)):
        axs[axs_ind].tick_params(axis='both', which='major', labelsize = 20)
        axs[axs_ind].tick_params(axis='both', which='minor', labelsize = 16)
    
    #plot acc
    axs[0].plot(history.history['accuracy'])
    axs[0].set_ylabel('Accuracy', fontsize = 20.0)
    
    #plot TPR
    axs[1].plot(history.history['TPR'])
    axs[1].set_ylabel('TPR', fontsize = 20.0)

    #plot TNR    
    axs[2].plot(history.history['TNR'])
    axs[2].set_ylabel('TNR', fontsize = 20.0)
    axs[2].set_xlabel('Epochs', fontsize = 20.0)
    
    fig.tight_layout()
    plt.savefig(join(save_path, 'Metrics.png'))
    plt.close()
    

def quantize_model(model, params, quant_ints):
    #import
    from cnn2snn import quantize, quantize_layer
    
    #quantize network
    model_quantized = quantize(model, input_weight_quantization = quant_ints[0],
                             weight_quantization = quant_ints[1],
                             activ_quantization = quant_ints[2])
    #quantize output layer to a lower bitwdith
    layer_name = model_quantized.layers[-1].name
    model_quantized = quantize_layer(model_quantized, layer_name, bitwidth = 4)
        
    #compile model
    from models_exp3 import compile_model
    model_quantized, custom_metrics = compile_model(model_quantized)
    
    return model_quantized
    
    
def estimate_spikes(net, dataset):
    import numpy as np
    
    #estimate average number of spikes produced by pre-trained layers
    for step, (inputs, targets) in enumerate(dataset):
        spikes = net.forward(inputs.numpy())
        
        #gather em up
        if step == 0:
            total_spikes = spikes.sum(axis = (1,2,3))
        else:
            total_spikes = np.concatenate([total_spikes, spikes.sum(axis = (1,2,3))],
                                                                    axis = 0)
            
    #compute average number of spikes
    spike_ave = np.mean(total_spikes)
    
    return spike_ave

def estimate_num_neurons(net, dataset, params):
    from akida import AkidaUnsupervised
    from akida import Model, InputData, FullyConnected
    from tools_exp3 import plot_neurons_per_class_vs_loss, plot_learned_neurons_vs_num_neurons
    import numpy as np
    
    def compute_losses(net, spikes, neurons_per_class, num_weights,
                       num_classes = 2, initial_plasticity = 1, min_plasticity = 0.1,
                       plasticity_decay = 0.25,
                       learning_competition=0.0, num_repetitions=1):
        
        def create_one_fc_model(units):
            
            model_fc = Model()
            model_fc.add(InputData(name="input",
                                   input_shape = (spikes.shape[-3],
                                                  spikes.shape[-2],
                                                  spikes.shape[-1]),
                                   input_bits=1))
            layer_fc = FullyConnected(name='akida_edge_layer',
                                      units=units,
                                      activation=False)
            model_fc.add(layer_fc)
            model_fc.compile(optimizer=AkidaUnsupervised(num_weights=num_weights,
                                                         num_classes = num_classes,
                                                         initial_plasticity = initial_plasticity,
                                                         min_plasticity = min_plasticity,
                                                         plasticity_decay = plasticity_decay,
                                                         learning_competition=learning_competition))
            return model_fc
        
        losses = np.zeros((len(neurons_per_class), num_repetitions))
        num_learned_neurons = np.zeros((len(neurons_per_class), num_repetitions))
        for idx, n in enumerate(neurons_per_class):
            for i in range(num_repetitions):
                model_fc = create_one_fc_model(units = num_classes * n)
                
                #train model
                permut_spikes = np.random.permutation(spikes)
                model_fc.fit(permut_spikes)
                
                #get max potentials
                max_potentials = model_fc.forward(permut_spikes).max(axis = -1)
                losses[idx, i] = np.sum(num_weights - max_potentials)
                
                #get threshold learning
                th_learn = model_fc.get_layer('akida_edge_layer').get_variable('threshold_learning')
                num_learned_neurons[idx, i] = np.sum(th_learn > 0)
                
        return losses.min(axis=1)/len(spikes), num_learned_neurons.min(axis = 1)
    
    #estimate on dataset
    dataset_length = params.get('dataset_length')
    neurons_per_class = params.get('neurons_per_class')
    num_weights = params.get('num_weights')
    savepath = params.get('save_path')
    weights_mults = params.get('weights_mults')
    spike_est = params.get('spike_est')
    initial_plasticity = params.get('initial_plasticity')
    min_plasticity = params.get('min_plasticity')
    plasticity_decay = params.get('plasticity_decay')
    dataset = dataset.batch(batch_size = dataset_length)
    
    for weights_mult in weights_mults:
        num_weights = int(weights_mult * spike_est)
        end_str = f'weight_mult_{weights_mult}'
        
        for step, (inputs, targets) in enumerate(dataset):
            
            spikes = net.forward(inputs.numpy())
        
            losses, num_learned_neurons = compute_losses(net, spikes,
                                                         neurons_per_class,
                                                         num_weights,
                                                         initial_plasticity = initial_plasticity,
                                                         min_plasticity = min_plasticity,
                                                         plasticity_decay = plasticity_decay)
            
        #plot
        plot_neurons_per_class_vs_loss(neurons_per_class, losses, savepath, end_str)
        plot_learned_neurons_vs_num_neurons(neurons_per_class, num_learned_neurons,
                                            savepath, end_str)
        
        
def get_prelim_performance(net, dataset, paths):
    
    import numpy as np
    res_file_path = paths.get('res_file_path')
    
    #get preliminary performance on held out data
    for step, (inputs, targets) in enumerate(dataset):
        
        #check metrics
        pred = net.predict_classes(inputs.numpy(), num_classes = 2)
        
        if step == 0:
            total_pred = pred
            total_targ = targets.numpy()
        else:
            total_pred = np.concatenate([total_pred, pred], axis = 0)
            total_targ = np.concatenate([total_targ, targets.numpy()], axis = 0)
            
    #calculate metrics
    total_pred_bool = total_pred.astype(np.bool_)
    total_targ_bool = total_targ.astype(np.bool_)
    #acc
    acc = sum((total_pred_bool == total_targ_bool).astype(np.int8))/len(total_targ_bool); 
    #TPR
    TPR = (sum(np.logical_and(total_pred_bool, total_targ_bool).astype(np.int8)))/(sum(total_targ_bool.astype(np.int8)))
    #TNR
    TNR = (sum(np.logical_and(np.equal(total_pred_bool, False), np.equal(total_targ_bool, False)).astype(np.int8)))/(sum(np.equal(total_targ_bool, False).astype(np.int8)))
    res_file = open(res_file_path, 'a')
    res_file.write(f'Preliminary Acc: {acc}\n')
    res_file.write(f'Preliminary TPR: {TPR}\n')
    res_file.write(f'Preliminary TNR: {TNR}\n')
    res_file.write('\n')
    res_file.close()
    
    return [acc, TPR, TNR]

def edge_learning(net, dataset, params, paths):
    
    import numpy as np
    import akida
    
    if not params.get('debug') and not params.get('run_cpu'):
        #map network to device
        devices = akida.devices()
        print(f'Available devices: {[dev.desc for dev in devices]}')
        device = devices[0]
        net.map(device)
    
    acc_per_epoch = []
    TNR_per_epoch = []
    TPR_per_epoch = []
    res_file_path = paths.get('res_file_path')
    res_file = open(res_file_path, 'a')
    print('edge learning starting')
    for epoch in range(params.get('max_epochs')):
        print(f'epoch: {epoch + 1}')
        res_file.write(f'Edge Learning Epoch: {epoch+1}\n')
        for step, (inputs, targets) in enumerate(dataset):
            #train
            net.fit(inputs.numpy(), targets.numpy())
            
            #get predictions
            pred = net.predict_classes(inputs.numpy(), num_classes = 2)
            
            if step == 0:
                total_pred = pred
                total_targ = targets.numpy()
            else:
                total_pred = np.concatenate([total_pred, pred], axis = 0)
                total_targ = np.concatenate([total_targ, targets.numpy()], axis = 0)
                
        #calculate metrics
        total_pred_bool = total_pred.astype(np.bool_)
        total_targ_bool = total_targ.astype(np.bool_)
        #acc
        acc = sum((total_pred_bool == total_targ_bool).astype(np.int8))/len(total_targ_bool); 
        #TPR
        TPR = (sum(np.logical_and(total_pred_bool, total_targ_bool).astype(np.int8)))/(sum(total_targ_bool.astype(np.int8)))
        #TNR
        TNR = (sum(np.logical_and(np.equal(total_pred_bool, False), np.equal(total_targ_bool, False)).astype(np.int8)))/(sum(np.equal(total_targ_bool, False)
                                                                                                                             .astype(np.int8)))
        
        #record
        res_file.write(f'Acc: {acc}\n')
        res_file.write(f'TPR: {TPR}\n')
        res_file.write(f'TNR: {TNR}\n')
        res_file.write('\n')
        
        #append
        acc_per_epoch.append(acc)
        TPR_per_epoch.append(TPR)
        TNR_per_epoch.append(TNR)
        
    res_file.close()
    
    return net, [acc_per_epoch, TPR_per_epoch, TNR_per_epoch]


def plot_metrics(metric_list, save_path):
    
    import matplotlib.pyplot as plt
    from os.path import join
    
    #create subplots
    fig, axs = plt.subplots(3)
    for axs_ind in range(len(axs)):
        axs[axs_ind].tick_params(axis='both', which='major', labelsize = 20)
        axs[axs_ind].tick_params(axis='both', which='minor', labelsize = 16)
    
    #plot acc
    axs[0].plot(metric_list[0])
    axs[0].set_ylabel('Accuracy', fontsize = 20.0)
    
    #plot TPR
    axs[1].plot(metric_list[1])
    axs[1].set_ylabel('TPR', fontsize = 20.0)
    
    #plot TNR
    axs[2].plot(metric_list[2])
    axs[2].set_ylabel('TNR', fontsize = 20.0)
    axs[2].set_xlabel('Epochs', fontsize = 20.0)
    
    fig.tight_layout()
    plt.savefig(join(save_path, 'Edge_Learning_Metrics.png'))
    plt.close()
    
    
    
def get_inf_times_and_energy(nets, dataset, paths, save_path = None):
    
    import akida
    from pyJoules.energy_meter import measure_energy
    from pyJoules.handler.csv_handler import CSVHandler
    import time
    from os.path import join, exists
    from os import remove
    import csv
    
    if save_path is None:
        results_path = paths.get('results_path')
    else:
        results_path = save_path
    
    #define csv path and remove old csv if it exists
    csv_path = join(results_path, 'pyJoules_energy_measurement_results.csv')
    if exists(csv_path):
        remove(csv_path)
        
    #define csv handler
    csv_handler = CSVHandler(csv_path)
    
    #define function for keras model
    @measure_energy(handler = csv_handler)
    def get_net_predictions(net, inputs):
                    
            pred = net(inputs)

            return pred
    
    #define function for getting a baseline energy reading
    @measure_energy(handler = csv_handler)
    def get_baseline_energy():
        
        time.sleep(1)
        done = 1
        
        return done
    
    #for net in nets, get network total epoch inf times and energy consumption for dataset
    net_count = -1
    for net in nets:
        net_count += 1
        
        #expecting first net to be the akida net
        if net_count == 0:
            #map net to device
            devices = akida.devices()
            device = devices[0]
            net.map(device)
    
            #enable power measurement
            device.soc.power_measurement_enabled = True
            
            inf_times = []
            framerates = []
            energy_readings = []
            power_readings = []
            
        else:
            #measure baseline energy for pyJoules
            done = get_baseline_energy()
        
        for step, (inputs, targets) in enumerate(dataset):
            
            #get inference time
            if net_count == 0:
                #time net inference
                time_start = time.time()
                pred = net.predict_classes(inputs.numpy(), num_classes = 2)
                time_end = time.time()
                
                #record time
                inf_time_batch = time_end - time_start
                inf_times.append(inf_time_batch)
                
                #record energy
                akida_stats = net.statistics
                akida_framerate = akida_stats.fps
                framerates.append(akida_framerate)
                akida_mJpf = akida_stats.energy
                akida_power_mW = akida_mJpf * akida_framerate
                power_readings.append(akida_power_mW)
                akida_energy_mJ = akida_power_mW * inf_time_batch
                energy_readings.append(akida_energy_mJ)
            
            #get energy if keras model
            else:
                pred = get_net_predictions(net, inputs)
           
        if net_count == 0:    
            inf_energy_epoch = sum(energy_readings)
            inf_time_epoch = sum(inf_times)
            
        else:
            
            #save data
            csv_handler.save_data()
            
            #read in csv that i just saved
            energy_used = []
            durations = []
            count = -1
            with open(csv_path, newline = '') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    count += 1
                    if count == 0:
                        continue
                    splits = row[0].split(';')
                    energy_used.append(eval(splits[-2]))
                    durations.append(eval(splits[-3]))
                    
            #get baseline power in mW
            baseline_power = energy_used[0] / 1000 / durations[0]
            # get epoch inf power
            inf_time_epoch = sum(durations[1:])
            inf_power_epoch = sum(energy_used[1:]) / 1000 / inf_time_epoch
            # subtract baseline power
            inf_power_epoch_adjusted = inf_power_epoch - baseline_power
            # get actual energy consumption in milli joules
            inf_energy_epoch = inf_power_epoch_adjusted * inf_time_epoch
              
        #record epoch time and energy for akida
        if net_count == 0:
            results = {'akida_time_epoch':inf_time_epoch,
                       'akida_energy_epoch':inf_energy_epoch}
        else:
            results.update({'CPU_time_epoch':inf_time_epoch,
                       'CPU_energy_epoch_adjusted':inf_energy_epoch,
                       'CPU_energy_epoch_total':inf_power_epoch * inf_time_epoch,
                       'CPU_energy_epoch_baseline':baseline_power * inf_time_epoch})
        
    return results