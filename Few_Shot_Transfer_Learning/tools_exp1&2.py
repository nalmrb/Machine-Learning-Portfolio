# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 06:52:56 2023

@author: nalmrb
"""

### Akida project functions

def load_data(filepath, params):
    from tensorflow.data import Dataset
    from os.path import join, isdir
    from os import listdir
    from glob import glob
    from numpy import transpose, concatenate, reshape, zeros, shape, array, load
    from numpy import uint8, int32, delete, pad
    from sklearn.preprocessing import minmax_scale
    from tools import fix_imbalanced_dataset
    from re import findall
    
    #unpack params
    chanInds = params.get('chanInds', list(range(19)))
    stage = params.get('stage')
    one_hot = params.get('one_hot')
    num_train_drivers = params.get('num_train_drivers')
    from_folders = params.get('from_folders', False)
    fix_imbalance = params.get('fix_imbalance', False)
    use_exp1_data = params.get('use_exp1_data', False)
    if not use_exp1_data:
        bootstrap_idx = params.get('bootstrap_idx')
        #define bootstrap driver data permutations
        bootstrap_permuts = [[3,5,10,12,18,20,22,24,14,16,26],
                             [12,18,20,22,24,14,16,26,3,5,10],
                             [22,24,14,16,26,3,5,10,12,18,20],
                             [16,26,3,5,10,12,18,20,22,24,14],
                             [5,10,12,18,20,22,24,14,16,26,3],
                             [18,20,22,24,14,16,26,3,5,10,12],
                             [24,14,16,26,3,5,10,12,18,20,22],
                             [26,3,5,10,12,18,20,22,24,14,16],
                             [10,12,18,20,22,24,14,16,26,3,5],
                             [20,22,24,14,16,26,3,5,10,12,18]]
    
    else:
        bootstrap_idx = params.get('bootstrap_idx')
        #overwrite bootstrap permuts
        bootstrap_permuts = [[7,8,9,11,13,15,17,1,2,4,6],
                              [1,2,4,6,7,8,9,11,13,15,17],
                              [11,13,15,17,1,2,4,6,7,8,9],
                              [6,7,8,9,11,13,15,17,1,2,4],
                              [17,1,2,4,6,7,8,9,11,13,15],
                              [9,11,13,15,17,1,2,4,6,7,8],
                              [4,6,7,8,9,11,13,15,17,1,2],
                              [15,17,1,2,4,6,7,8,9,11,13],
                              [8,9,11,13,15,17,1,2,4,6,7],
                              [2,4,6,7,8,9,11,13,15,17,1]]
        
        # [[7,8,9,11,13,15,17,19,21,23,25,1,2,4,6],
        #                      [1,2,4,6,7,8,9,11,13,15,17,19,21,23,25],
        #                      [13,15,17,19,21,23,25,1,2,4,6,7,8,9,11],
        #                      [21,23,25,1,2,4,6,7,8,9,11,13,15,17,19],
        #                      [2,4,6,7,8,9,11,13,15,17,19,21,23,25,1],
        #                      [8,9,11,13,15,17,19,21,23,25,1,2,4,6,7],
        #                      [15,17,19,21,23,25,1,2,4,6,7,8,9,11,13],
        #                      [23,25,1,2,4,6,7,8,9,11,13,15,17,19,21],
        #                      [4,6,7,8,9,11,13,15,17,19,21,23,25,1,2],
        #                      [9,11,13,15,17,19,21,23,25,1,2,4,6,7,8]]
            
    if stage == 'few_shot_learning':
        datasets = []
        dataset_lengths = []
        from tools import augment_data   
        num_iter = 5
    else:
        num_iter = 1
    
    #get directory contents
    if from_folders is True:
        filePathList = []
        folders = listdir(filepath)
        
        if stage == 'initial_training' or stage == 'quantized_retraining':
            load_folder_nums = bootstrap_permuts[bootstrap_idx][0:num_train_drivers]
        elif stage == 'few_shot_learning' or stage == 'get_inf_times':
            load_folder_nums = bootstrap_permuts[bootstrap_idx][num_train_drivers-len(bootstrap_permuts[0]):]
            
        for this_dir in folders:
            if isdir(join(filepath, this_dir)):
                #get number
                driver_num = findall(r'\d+',this_dir)
                if eval(driver_num[0]) in load_folder_nums:
                    filePathList.append(join(filepath, this_dir))
                        
    else:
        filePathList = [filepath]
        
    
    #define storage
    data_list = []
    label_list = []
    
    #breakpoint()
    print(filePathList)
    for this_filePath in filePathList:

        #get contents of directory
        fileList = glob(join(this_filePath, '*.npy'))

        #loop through data
        for fileInd in range(len(fileList)):
            for data_iter in range(num_iter):
                # read data from file
                data = load(fileList[fileInd])
                #drop channels not used
                if len(chanInds)<19:
                    chanToDel = [x for x in list(range(19)) if x not in chanInds]
                    data = delete(data, chanToDel, axis = 1)
                        
                #transform data
                if (996 - shape(data)[0]) < 0:
                    continue
                else:
                    data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                         shape(data)[1]])],0))
                if data_iter > 0:
                    data = augment_data(data)
                data = minmax_scale(data, axis = 1)
                #convert to 8bit integer
                data = data * 255
                data = reshape(data, [shape(data)[0], shape(data)[1],1])
                data = data.astype(uint8)
                #data = data.astype(float64)
                #add back data as zeros if channels were deleted
                for chan_idx in range(19):
                    if chan_idx not in chanInds:
                        data = pad(data, ((0,1),(0,0),(0,0)))
                
                #save data to list
                data_list.append(data)
                
                #get label
                if fileList[fileInd][-7:-4] == '1-0':
                    if one_hot:
                        label = array([0,1], dtype = int32)
                    else:
                        label = int32(1.)
                else:
                    if one_hot:
                        label = array([1,0], dtype = int32)
                    else:
                        label = int32(0.)
                
                #save label to list
                label_list.append(label)
            
            #stop short if debug
            if params.get('debug'):
                if len(data_list) >= 50:
                    break
            
        if stage == 'few_shot_learning':
            #correct and save driver data individually
            data_list, label_list = fix_imbalanced_dataset(data_list, label_list)
            
            # change into tensorflow dataset
            data_array = array(data_list)
            label_array = array(label_list)
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

    #multiply number of positive examples by 4 if fix_imbalance = True      
    if not stage == 'few_shot_learning':
        if fix_imbalance:
            data_list, label_list = fix_imbalanced_dataset(data_list, label_list)
        # change into tensorflow dataset
        data_array = array(data_list)
        label_array = array(label_list)
        dataset = Dataset.from_tensor_slices((data_array, label_array))            
        
        print(f'dataset length: {len(data_list)}')   
        return dataset, len(data_list)
    else:
        print(f'dataset length: {dataset_lengths}')
        return datasets, dataset_lengths

# fix imbalanced dataset
def fix_imbalanced_dataset(data_list, label_list):
    
    #get all positive label indices
    Pos_ind  = [index for (index, item) in enumerate(label_list) if item == 1]
    #sort data into positive and negatives
    P_data = []
    P_labels = []
    N_data = []
    N_labels = []
    for idx in range(len(label_list)):
        if idx in Pos_ind:
            P_data.append(data_list[idx])
            P_labels.append(label_list[idx])
        else:
            N_data.append(data_list[idx])
            N_labels.append(label_list[idx])
    #class imbalance is approximately 4 to one so duplicate these lists 4 times
    P_data_dup = P_data * 4
    P_labels_dup = P_labels * 4
    #append to negatives
    data_list = N_data + P_data_dup
    label_list = N_labels + P_labels_dup
    
    return data_list, label_list

def augment_data(data):
    import numpy as np
            
    #get shape of data sample
    data_shape = np.shape(data)
    
    #create gaussian noise
    noise = np.random.normal(0, 1, data_shape)
    
    #add to data
    aug_data = data + noise
    
    return aug_data

#get split dataset
def get_split_dataset(ds, indices):
    import tensorflow as tf
    
    x_indices = tf.constant(indices, dtype = tf.int64)
    
    def is_index_in(index, rest):
        # Returns True if the specified index value is included in X_indices_ts.
        #
        # '==' compares the specified index value with each values in X_indices_ts.
        # The result is a boolean tensor, looks like [ False, True, ..., False ].
        # reduce_any() returns Ture if True is included in the specified tensor.
        return tf.math.reduce_any(index == x_indices)
    
    def drop_index(index, rest):
        return rest

    # Dataset.enumerate() is similter to Python's enumerate().
    # The method adds indices to each elements. Then, the elements are filtered
    # by using the specified indices. Finally unnecessary indices are dropped.
    selected_ds = ds \
        .enumerate() \
        .filter(is_index_in) \
        .map(drop_index)
    return selected_ds

#calculate class weights
def calculate_class_weights(dataset, one_hot):
    import numpy as np
    
    y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                        dtype = object)
    y_targ_shape = np.shape(y_targets)
    if len(y_targ_shape) > 2:
        y_targets = np.reshape(y_targets,
                               newshape = [y_targ_shape[-2], y_targ_shape[-1]])
    sum1 = 0
    sum2 = 0
    for y in y_targets:
        if one_hot:
            if y[1] == 1:
                sum2 += 1
            else:
                sum1 += 1
        else:
            if y == 1:
                sum2 += 1
            else:
                sum1 += 1
    if sum1 > 0:        
      classWeight1 = len(y_targets)/(2*sum1)
    else:
      classWeight1 = 0
    if sum2 > 0:
      classWeight2 = len(y_targets)/(2*sum2)
    else:
      classWeight2 = 0
    class_weights = [classWeight1, classWeight2]
    
    return class_weights

def get_dataset(params):
    
    import numpy as np
    from tools import calculate_class_weights
    from tools import get_split_dataset
    from sklearn.model_selection import train_test_split
    from tools import load_data
    
    #unpack params
    stage = params.get('stage')
    shuffle_buffer_size = params.get('shuffle_buffer_size')
    batch_size = params.get('batch_size')
    data_filepath = params.get('data_filepath')
    one_hot = params.get('one_hot')
    split_data = params.get('split_data', False)
    
    if stage == 'baseline':
        dataset, dataset_length = load_data(data_filepath, params)    
    elif (stage == 'initial_training' or stage == 'quantized_retraining'):
        params.update({'from_folders':True})
        dataset, dataset_length = load_data(data_filepath, params)
    elif (stage == 'few_shot_learning'):
        params.update({'from_folders':True})
        params.update({'fix_imbalance':True})
        datasets, dataset_lengths = load_data(data_filepath, params)

    elif (stage == 'get_inf_times'):
        params.update({'from_folders':True})
        params.update({'fix_imbalance':False})
        dataset, dataset_length = load_data(data_filepath, params)
        
        
    #get class weights
    if stage != "few_shot_learning":
        class_weights = calculate_class_weights(dataset, one_hot)
    else:
        class_weights = []
    
    #get split indices
    if split_data:
        #create train test split
        #get labels
        y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                             dtype = object)
        x_indices = np.arange(len(y_targets))
        
        train_split_frac = 0.8
        x_tr_ind, x_te_ind, y_tr_ind, y_te_ind = train_test_split(x_indices,
                                                                  y_targets,
                                                                  test_size = (1-train_split_frac),
                                                                  stratify = y_targets,
                                                                  random_state = 42)
        
        train_dataset = get_split_dataset(dataset, x_tr_ind)
        train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
        
        test_dataset = get_split_dataset(dataset, x_te_ind)
        test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
        
        datasets = [train_dataset, test_dataset]
        dataset_lengths = {'train':len(x_tr_ind), 'test':len(x_te_ind)} 
    else:
        if "datasets" in locals():
            for dataset_idx in range(len(datasets)):
                if batch_size == -1:
                    batch_size = dataset_lengths[dataset_idx]
                datasets[dataset_idx] = datasets[dataset_idx].shuffle(shuffle_buffer_size).batch(batch_size)
        else:
            if batch_size == -1:
                batch_size = dataset_length
            datasets = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
            dataset_lengths = dataset_length
          
        
    return datasets, dataset_lengths, class_weights

def Sup_Learning_EEG_Data_Keras(paths, params, net = None):
    
    import tensorflow as tf
    import keras
    from models import build_and_compile_CNN
    from tools import get_dataset
    from os.path import join 
    
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #define params
    batch_size = params.get('batch_size')
    max_epochs = params.get('max_epochs')
    network = params.get('network')
    split_dataset = params.get('split_dataset')
    if split_dataset is None:
        split_dataset = False
    
    #define dataset
    datasets, dataset_lengths, class_weights = get_dataset(params)
    
    if split_dataset:
        train_dataset = datasets[0]
        test_dataset = datasets[1]
    else:
        train_dataset = datasets
    
    #build and compile network
    if net is None:
        with strategy.scope():
            if network == 'CNN':
                dummy_input_shape = [1,19,996,1]
                net, custom_metrics = build_and_compile_CNN(dummy_input_shape)
    else:
        custom_metrics = None
                
    # define some optional callbacks for debugging
    # class get_metric_states(tf.keras.callbacks.Callback):
    #     def __init__(self, net):
    #         super().__init__()
    #         self.TPR_object = net.compiled_metrics._metrics[-2]
    #         self.TNR_object = net.compiled_metrics._metrics[-1]
        
    #     # def on_epoch_begin(self, epoch, logs=None):
    #     #     print("At Start of Epoch:")
    #     #     print(f"Num True Pos: {self.TPR_object.true_positives.numpy()}")
    #     #     print(f"Num Total Pos: {self.TPR_object.total_positives.numpy()}")
    #     #     print(f"Pred class length: {self.TPR_object.pred_class_len.numpy()}")
    #     #     print(f"true_pos_val_len: {self.TPR_object.true_pos_val_len.numpy()}")
    #     #     print(f"Num True Neg: {self.TNR_object.true_negatives.numpy()}")
    #     #     print(f"Num Total Neg: {self.TNR_object.total_negatives.numpy()}")
    #     #     print(f"Pred class length: {self.TNR_object.pred_class_len.numpy()}")
    #     #     print(f"true_neg_val_len: {self.TNR_object.true_neg_val_len.numpy()}")
            
    #     def on_epoch_end(self, epoch, logs=None):
    #         print("At End of Epoch:")
    #         print(f'y_true_len: {self.TPR_object.y_true_len.numpy()}')
    #         print(f"Num True Pos: {self.TPR_object.true_positives.numpy()}")
    #         print(f"Num Total Pos: {self.TPR_object.total_positives.numpy()}")
    #         print(f"Pred class length: {self.TPR_object.pred_class_len.numpy()}")
    #         print(f'y_true_len: {self.TNR_object.y_true_len.numpy()}')
    #         print(f"Num True Neg: {self.TNR_object.true_negatives.numpy()}")
    #         print(f"Num Total Neg: {self.TNR_object.total_negatives.numpy()}")
    #         print(f"Pred class length: {self.TNR_object.pred_class_len.numpy()}")
            
    # get_metric_states_ob = get_metric_states(net = net)
    
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
    
    #define class weights and batch_size and model_path
    data_point = next(iter(train_dataset))
    if batch_size != tf.shape(data_point[0])[0]:
        batch_size = tf.shape(data_point[0])[0]
    class_weights_dict = {0:class_weights[0], 1:class_weights[1]}
    model_path = paths.get('models_path')
    
    #define model checkpoint callback
    chkpt_path = join(model_path, 'cp.ckpt')
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath = chkpt_path,
        monitor = 'loss',
        mode = 'min',
        save_best_only=True,
        save_weights_only=True)
    
    #define early stop based on loss callback (to save time)
    loss_early_stop_callback = keras.callbacks.EarlyStopping(
        monitor = "loss",
        min_delta = 1e-3,
        patience = 10,
        restore_best_weights = True,
        start_from_epoch = 30)
    
    
    #define custom callback object if quantize retrain
    if params.get('stage') == 'quantized_retraining':
        check_metric_scores_early_stop_obj = check_metric_scores_early_stop(metric_targ = params.get('metric_target'))
        call_backs = [check_metric_scores_early_stop_obj]
    elif params.get('stage') == 'initial_training':
        call_backs = [model_checkpoint_callback]
        
    #train
    train_hist = net.fit(train_dataset, epochs = max_epochs, batch_size = batch_size,
            class_weight = class_weights_dict,
            callbacks = call_backs)
    # if params.get('stage') == 'initial_training':
    #     #load back best weights
    #     net.load_weights(chkpt_path)
    
    #check final metrics    
    if split_dataset:
        #eval model
        eval_score = net.evaluate(test_dataset)
    else:
        eval_score = net.evaluate(train_dataset)
    
    #save model
    stage = params.get('stage')
    if stage == 'initial_training':
        net.save(join(model_path, 'keras_cnn_exp2.keras'))
    elif stage == 'quantized_retraining':
        net.save(join(model_path, 'keras_cnn_exp2_quantized.keras'))
    
    return net, train_hist, eval_score, custom_metrics


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
    
    #define dataset
    dataset, dataset_length, class_weights = get_dataset(params)
        
    #compile model
    from models import compile_model
    model_quantized, custom_metrics = compile_model(model_quantized)

    #check metrics
    score = model_quantized.evaluate(dataset)
    
    return model_quantized, score


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
    
def plot_neurons_per_class_vs_loss(neurons_per_class, losses, savepath, end_str):
    
    import matplotlib.pyplot as plt
    from os.path import join
    import numpy as np
    
    plt.plot(np.array(neurons_per_class), losses)
    plt.xlabel("Neurons per class")
    plt.ylabel("Loss")
    plt.title("Losses vs number of neurons")
    plt.grid(linestyle = '--')
    end_str = end_str.replace('.', 'p')
    plt.savefig(join(savepath, f'Loss_vs_num_neurons_{end_str}.png'))
    plt.close()
    
def plot_learned_neurons_vs_num_neurons(neurons_per_class, num_learned_neurons, savepath, end_str):
    
    import matplotlib.pyplot as plt
    from os.path import join
    import numpy as np
    
    plt.plot(np.array(neurons_per_class), num_learned_neurons)
    plt.xlabel("Number of neurons per class")
    plt.ylabel("Nb of neurons that learned")
    plt.title("Num neurons learned vs Num neurons")
    plt.grid(linestyle = '--')
    end_str = end_str.replace('.', 'p')
    plt.savefig(join(savepath, f'Num_neurons_learned_vs_num_neurons_{end_str}.png'))
    plt.close()
    
    
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
    from tools import plot_neurons_per_class_vs_loss, plot_learned_neurons_vs_num_neurons
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
        
        
def get_prelim_performance(net, dataset, params):
    
    import numpy as np
    res_file_path = params.get('res_file_path')
    
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
    
    
def edge_learning(net, dataset, params):
    import numpy as np
    import akida
    
    max_epochs = params.get('max_epochs')
    res_file_path = params.get('res_file_path')
    
    if not params.get('debug') and not params.get('run_cpu'):
        #map net to device
        devices = akida.devices()
        print(f'Available devices: {[dev.desc for dev in devices]}')
        device = devices[0]
        net.map(device)
    
    acc_per_epoch = []
    TNR_per_epoch = []
    TPR_per_epoch = []
    res_file = open(res_file_path, 'a')
    print('edge learning starting')
    for epoch in range(max_epochs):
        print(f'epoch: {epoch+1}')
        res_file.write(f'Edge Learning Epoch: {epoch+1}\n')
        for step, (inputs, targets) in enumerate(dataset):

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
        TNR = (sum(np.logical_and(np.equal(total_pred_bool, False), np.equal(total_targ_bool, False)).astype(np.int8)))/(sum(np.equal(total_targ_bool, False).astype(np.int8)))
        
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
            results = {f'akida_time_epoch':inf_time_epoch,
                       f'akida_energy_epoch':inf_energy_epoch}
        else:
            results.update({f'CPU_time_epoch':inf_time_epoch,
                       f'CPU_energy_epoch_adjusted':inf_energy_epoch,
                       f'CPU_energy_epoch_total':inf_power_epoch * inf_time_epoch,
                       f'CPU_energy_epoch_baseline':baseline_power * inf_time_epoch})
        
#    #single trial energy and time
#    net_count = -1
#    for net in nets:
#        net_count += 1
#        
#        for step, (inputs, targets) in enumerate(dataset):
#            
#            #get single trial
#            single_trial_input = inputs[0,:,:,:]
#            single_trial_input = expand_dims(single_trial_input, axis = 0)
#            
#            #get inference time
#            if net_count == 0:
#                #time prediction
#                time_start = time.time()
#                pred = net.predict_classes(single_trial_input.numpy(), num_classes = 2)
#                time_end = time.time()
#                
#                #record time
#                single_trial_time = time_end - time_start
#                
#                #record energy
#                akida_stats = net.statistics
#                breakpoint()
#                akida_framerate = akida_stats.fps
#                akida_mJpf = akida_stats.energy
#                akida_power_mW = akida_mJpf * akida_framerate
#                akida_energy_mJ = akida_power_mW * inf_time_batch
#                
#                #get time and energy for keras model
#            else: 
#                pred = get_net_predictions(net, inputs)
#                print(pred)
#                
#            break
#        
#        if net_count == 0:
#            single_trial_energy = akida_energy_mJ
#            
#        else:
#            
#            #save data
#            csv_handler.save_data()
#            
#            #read in csv that i just saved
#            energy_used = []
#            durations = []
#            count = -1
#            with open(csv_path, newline = '') as csvfile:
#                reader = csv.reader(csvfile)
#                for row in reader:
#                    count += 1
#                    if count == 0:
#                        continue
#                    splits = row[0].split(';')
#                    energy_used.append(eval(splits[-2]))
#                    durations.append(eval(splits[-3]))
#                    
#            #get baseline power in mW
#            baseline_power = energy_used[0] / 1000 / durations[0]
#            # get inf power
#            single_trial_time =  durations[-1]
#            inf_power = energy_used[-1] / 1000 / single_trial_time
#            #subtract baseline power
#            inf_power_actual = inf_power - baseline_power
#            #get actual energy consumption in milli joules
#            single_trial_energy = inf_power_actual * single_trial_time
#            
#        #record single trial results
#        results = {f'net_{net_count}_time_single_trial':single_trial_time,
#                   f'net_{net_count}_energy_single_trial':single_trial_energy}
        
    return results


def get_dataset_statistics(paths):
    
    import numpy as np
    from tools import get_dataset
    
    exp1_data_path = paths.get('exp1_train_data_path')
    bootstrap_data_path = paths.get('bootstrap_data_path')
    
    #get exp1 metrics
    params = {'stage':'baseline','shuffle_buffer_size':500,'batch_size':8,
              'data_filepath':exp1_data_path,'one_hot':True,'split_data':False}
    exp1_dataset, exp1_dataset_length, class_weights = get_dataset(params)
    print(f'Exp1 data points: {exp1_dataset_length}')
    num_positives = int(exp1_dataset_length/(2*class_weights[1]))
    num_negatives = int(exp1_dataset_length/(2*class_weights[0]))
    print(f'Exp1 Num Positives: {num_positives}')
    print(f'Exp1 Num Negatives: {num_negatives}')
    
    
    #define bootstrap driver data permutations
    bootstrap_permuts = [[3,5,10,12,18,20,22,24,14,16,26],
                         [12,18,20,22,24,14,16,26,3,5,10],
                         [22,24,14,16,26,3,5,10,12,18,20],
                         [16,26,3,5,10,12,18,20,22,24,14],
                         [5,10,12,18,20,22,24,14,16,26,3],
                         [18,20,22,24,14,16,26,3,5,10,12],
                         [24,14,16,26,3,5,10,12,18,20,22],
                         [26,3,5,10,12,18,20,22,24,14,16],
                         [10,12,18,20,22,24,14,16,26,3,5],
                         [20,22,24,14,16,26,3,5,10,12,18]]
    
    #loop through permutations and record stats
    dataset_metrics_hist = np.zeros([6,len(bootstrap_permuts)])
    for loop_idx in range(len(bootstrap_permuts)):
        print(f'loop number: {loop_idx}')
        params.update({'bootstrap_idx':loop_idx,'data_filepath':bootstrap_data_path})
        
        #get train set
        params.update({'stage':'initial_training'})
        
        dataset, dataset_length, class_weights = get_dataset(params)
        print('Train Dataset')
        print(f'Total data points: {dataset_length}')
        if class_weights[1] > 0:
            num_positives = int(dataset_length/(2*class_weights[1]))
        else:
            num_positives = 0
        if class_weights[0] > 0:
            num_negatives = int(dataset_length/(2*class_weights[0]))
        else:
            num_negatives = 0
        print(f'Positives: {num_positives}')
        print(f'Negatives: {num_negatives}')
        
        dataset_metrics_hist[0,loop_idx] = dataset_length
        dataset_metrics_hist[1,loop_idx] = num_positives
        dataset_metrics_hist[2,loop_idx] = num_negatives
        
        #get test set
        params.update({'stage':'get_inf_times'})
        
        dataset, dataset_length, class_weights = get_dataset(params)
        print('Test Dataset')
        print(f'Total data points: {dataset_length}')
        if class_weights[1] > 0:
            num_positives = int(dataset_length/(2*class_weights[1]))
        else:
            num_positives = 0
        if class_weights[0] > 0:
            num_negatives = int(dataset_length/(2*class_weights[0]))
        else:
            num_negatives = 0
        
        dataset_metrics_hist[3,loop_idx] = dataset_length
        dataset_metrics_hist[4,loop_idx] = num_positives
        dataset_metrics_hist[5,loop_idx] = num_negatives
        print(f'Positives: {num_positives}')
        print(f'Negatives: {num_negatives}')
        
    
    #calculate grand average and standard deviation
    grand_ave = np.mean(dataset_metrics_hist, axis = 1)
    std = np.std(dataset_metrics_hist, axis = 1)
    
    print(f'Train Set Total data points: {grand_ave[0]} ({std[0]})')
    print(f'Train set grand ave positives: {grand_ave[1]} ({std[1]})')
    print(f'Train set grand ave negatives: {grand_ave[2]} ({std[2]})')
    print(f'Test set Total data points: {grand_ave[3]} ({std[3]})')
    print(f'Test set grand ave positives: {grand_ave[4]} ({std[4]})')
    print(f'Test set grand ave negatives: {grand_ave[5]} ({std[5]})')
    
