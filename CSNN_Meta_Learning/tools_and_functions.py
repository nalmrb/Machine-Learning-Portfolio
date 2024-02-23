# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:00:34 2022

@author: nalmrb
"""

#tools and models

#import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
from os.path import isdir
from glob import glob
from numpy import transpose, concatenate, reshape, zeros, shape, array
from sklearn.preprocessing import minmax_scale

#data tools
#define data loading function or class
def Load_dataset(filePath, one_hot = False, debug = False):
    ## this function loads the experiment 1 data
    
    #create storage
    data_list = []
    label_list = []
    #get directory contents
    exp1_data = glob(join(filePath,'*.npy'))

    #loop through data
    if debug:
        count = 0
    for fileInd in range(len(exp1_data)):
        if debug:
            count += 1
            if count >= 101:
                break
        
        # read data from file
        data = np.load(exp1_data[fileInd])
        #transform data
        data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                 shape(data)[1]])],0))
        data = minmax_scale(data, axis = 1)
        data = reshape(data, [shape(data)[0], shape(data)[1],1])
        data = data.astype(np.float64)
        
        #save data to list
        data_list.append(data)
        
        #get label
        if exp1_data[fileInd][-7:-4] == '1-0':
            if one_hot:
                label = np.array([0,1])
            else:
                label = np.float32(1.)
        else:
            if one_hot:
                label = np.array([1,0])
            else:
                label = np.float32(0.)
        
        #save label to list
        label_list.append(label)
    
    # change into tensorflow dataset
    data_array = array(data_list)
    label_array = array(label_list)
    exp1_dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    
    return exp1_dataset, len(exp1_data)

def Load_all_driver_data(filepath, one_hot = False, debug = False):
    ## this function loads the data from each driver and returns one dataset
    #expecting path to folder housing driver data folders
        
    #define storage
    data_list = []
    label_list = []
    
    #get directory contents
    driver_dirs = listdir(filepath)
    
    for this_dir in driver_dirs:
        if isdir(join(filepath, this_dir)):

            #get contents of directory
            fileList = glob(join(filepath, this_dir, '*.npy'))

            #loop through data
            if debug:
                count = 1
            for fileInd in range(len(fileList)):
                if debug:
                    count += 1
                    if count >= 11:
                        break
                # read data from file
                data = np.load(fileList[fileInd])
                #transform data
                if (996 - shape(data)[0]) < 0:
                    continue
                else:
                    data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                         shape(data)[1]])],0))
                data = minmax_scale(data, axis = 1)
                data = reshape(data, [shape(data)[0], shape(data)[1],1])
                data = data.astype(np.float64)
                
                #save data to list
                data_list.append(data)
                
                #get label
                if fileList[fileInd][-7:-4] == '1-0':
                    if one_hot:
                        label = np.array([0,1])
                    else:
                        label = np.float32(1.)
                else:
                    if one_hot:
                        label = np.array([1,0])
                    else:
                        label = np.float32(0.)
                
                #save label to list
                label_list.append(label)
            
    # change into tensorflow dataset
    data_array = array(data_list)
    label_array = array(label_list)
    dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))            
    
    return dataset, len(fileList)
    
# load individual driver data
# this function expects a path to the specific driver data folder then returns that data in a dataset
def Load_driver_data(filepath, one_hot = False):
    #create storage
    data_list = []
    label_list = []
    #get contents of directory
    fileList = glob(join(filepath,'*.npy'))

    #loop through data
    for fileInd in range(len(fileList)):
        # read data from file
        data = np.load(fileList[fileInd])
        #transform data
        if (996 - shape(data)[0]) < 0:
            continue
        else:
            data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                 shape(data)[1]])],0))
        data = minmax_scale(data, axis = 1)
        data = reshape(data, [shape(data)[0], shape(data)[1],1])
        data = data.astype(np.float64)
        
        #save data to list
        data_list.append(data)
            
        #get label
        if fileList[fileInd][-7:-4] == '1-0':
            if one_hot:
                label = np.array([0,1])
            else:
                label = np.float32(1.)
        else:
            if one_hot:
                label = np.array([1,0])
            else:
                label = np.float32(0.)
        
        #save label to list
        label_list.append(label)
    
    # change into tensorflow dataset
    data_array = array(data_list)
    label_array = array(label_list)
    this_driver_dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    
    return this_driver_dataset, len(fileList)

def Load_driver_data_random_subset(filepath, count, one_hot = False):
    from random import sample
    #create storage
    data_list = []
    label_list = []
    #get contents of directory
    fileList = glob(join(filepath,'*.npy'))

    #loop through data
    for fileInd in range(len(fileList)):
        # read data from file
        data = np.load(fileList[fileInd])
        #transform data
        if (996 - shape(data)[0]) < 0:
            continue
        else:
            data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                 shape(data)[1]])],0))
        data = minmax_scale(data, axis = 1)
        data = reshape(data, [shape(data)[0], shape(data)[1],1])
        data = data.astype(np.float64)
        
        #save data to list
        data_list.append(data)
            
        #get label
        if fileList[fileInd][-7:-4] == '1-0':
            if one_hot:
                label = np.array([0,1])
            else:
                label = np.float32(1.)
        else:
            if one_hot:
                label = np.array([1,0])
            else:
                label = np.float32(0.)
        
        #save label to list
        label_list.append(label)
    
    #get arrays of indices where label equals 0 or 1
    if one_hot:
        label_array_0_ind = [index for (index,item) in enumerate(label_list) if item[1] == 0]
        label_array_1_ind = [index for (index,item) in enumerate(label_list) if item[1] == 1]
    else:
        label_array_0_ind = [index for (index,item) in enumerate(label_list) if item == 0]
        label_array_1_ind = [index for (index,item) in enumerate(label_list) if item == 1]
    #get half count
    half_count_0 = int(np.floor(count/2))
    half_count_1 = int(np.ceil(count/2))
    #get random subset of index arrays according to count
    zero_ind_subset = sample(label_array_0_ind, half_count_0)
    one_ind_subset = sample(label_array_1_ind, half_count_1)
    #sample from data or label
    indices_list = zero_ind_subset + one_ind_subset
    data_array = array([data_list[i] for i in indices_list])
    label_array = array([label_list[i] for i in indices_list])
    #create tensorflow dataset
    this_driver_dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    
    return this_driver_dataset, len(zero_ind_subset) + len(one_ind_subset)

def load_driver_data_in_order(filepath, number_of_trials=0, one_hot = False):
    from random import shuffle
    
    #create storage
    data_list = []
    label_list = []
    ordered_fileList = []
    #get contents of directory
    fileList = glob(join(filepath,'*.npy'))
    #randomize fileList
    shuffle(fileList)
    
    trials_loaded = 0
    #pull out all data of full trials, order then store
    while len(fileList) >= 1:
        if number_of_trials > 0 and trials_loaded == number_of_trials:
            break
        else:
            #trial to match
            splits = fileList[0].split('\\')
            trial_num = splits[-1][0:-9]
            
            #get all trials corresponding to trial number
            files_with_trial_num = []
            for file in fileList:
                if trial_num in file:
                    files_with_trial_num.append(file)
                    
            #if a complete trial exists
            if len(files_with_trial_num) == 5:
                #order
                indices = [0,1,2,3,4]
                for file in files_with_trial_num:
                    if '5-4' in file:
                        indices[0] = fileList.index(file)
                    elif '4-3' in file:
                        indices[1] = fileList.index(file)
                    elif '3-2' in file:
                        indices[2] = fileList.index(file)
                    elif '2-1' in file:
                        indices[3] = fileList.index(file)
                    else:
                        indices[4] = fileList.index(file)
                        
                #loop through indices and load and store data
                for fileInd in indices:
                    # read data from file
                    data = np.load(fileList[fileInd])
                    #transform data
                    data = transpose(concatenate([data,zeros([996-shape(data)[0],
                                                             shape(data)[1]])],0))
                    data = minmax_scale(data, axis = 1)
                    data = reshape(data, [shape(data)[0], shape(data)[1],1])
                    data = data.astype(np.float64)
                    
                    #save data to list
                    data_list.append(data)
                    
                    #get label
                    if fileList[fileInd][-7:-4] == '1-0':
                        if one_hot:
                            label = np.array([0,1])
                        else:
                            label = np.float32(1.)
                    else:
                        if one_hot:
                            label = np.array([1,0])
                        else:
                            label = np.float32(0.)
                    
                    #save label to list
                    label_list.append(label)
                    
                    #save filename to ordered file list
                    ordered_fileList.append(fileList[fileInd])
                
                #increment trials loaded
                trials_loaded += 1
                    
            #delete files from fileList
            for file in files_with_trial_num:
                fileList.remove(file)
            
    # change into tensorflow dataset
    data_array = array(data_list)
    label_array = array(label_list)
    this_driver_dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))

    return this_driver_dataset, len(ordered_fileList), ordered_fileList

def load_driver_data_in_order_window(filepath, window_size = 1, count = 9999,
                                    one_hot = False):
    from random import shuffle
    
    #define pad_to_size
    pad_to_size = 996 * window_size # 996 determined to be the longest data point in the 1st experiment data
    
    #create storage
    data_list = []
    label_list = []
    #get contents of directory
    fileList = glob(join(filepath,'*.npy'))
    #randomize fileList
    shuffle(fileList)
    
    data_points_added = 0
    while data_points_added < count and len(fileList) >= 1:
        #trial to match
        splits = fileList[0].split('\\')
        trial_num = splits[-1][0:-9]
        
        #get all trials corresponding to trial number
        files_with_trial_num = []
        for file in fileList:
            if trial_num in file:
                files_with_trial_num.append(file)
        
        #if a complete trial exists
        if len(files_with_trial_num) == 5:
            #order
            indices = [0,1,2,3,4]
            for file in files_with_trial_num:
                if '5-4' in file:
                    indices[0] = fileList.index(file)
                elif '4-3' in file:
                    indices[1] = fileList.index(file)
                elif '3-2' in file:
                    indices[2] = fileList.index(file)
                elif '2-1' in file:
                    indices[3] = fileList.index(file)
                else:
                    indices[4] = fileList.index(file)
        
            #loop through indices and load data
            data_temp_list = []
            for fileInd in indices:
                # read data from file
                data = np.load(fileList[fileInd])
                data = transpose(data)
                
                data_temp_list.append(data)
        
            #create data_points with size window_size
            for ii in range(len(data_temp_list)- window_size + 1):
                for jj in range(window_size):
                    if jj == 0:
                        data_point = data_temp_list[ii]
                    else:
                        data_point = concatenate([data_point, data_temp_list[ii+jj]], axis = 1)
                        
                #complete processing
                data_point = concatenate([data_point,zeros([shape(data_point)[0],
                                                pad_to_size-shape(data_point)[1]])],1)
                data_point = minmax_scale(data_point, axis = 1)
                data_point = reshape(data_point, [shape(data_point)[0],
                                                  shape(data_point)[1],1])
                data_point = data_point.astype(np.float64)
    
                #add to list
                data_list.append(data_point)
                data_points_added += 1
                
                #add label
                if one_hot:
                    if (ii + jj) == 4:
                        label_list.append(np.array([0,1]))
                    else:
                        label_list.append(np.array([1,0]))
                else:
                    if (ii + jj) == 4:
                        label_list.append(np.float32(1.))
                    else:
                        label_list.append(np.float32(0.))
                        
                if data_points_added >= count:
                    break
       
        #delete files from fileList
        for file in files_with_trial_num:
            fileList.remove(file)     
        
    # change into tensorflow dataset
    data_array = array(data_list)
    label_array = array(label_list)
    this_driver_dataset = tf.data.Dataset.from_tensor_slices((data_array, label_array))
    
    return this_driver_dataset, len(data_array)


def Load_LSG_Supervised_Learning_Dataset(filePath):
    ## this function loads in the LSG input and learning signals to train the LSG
    from tools_and_functions import load_data
    
    #create storage
    total_inputs = None
    total_learning_signals = None
    
    #get input contents
    input_filepath = join(filePath, 'inputs')
    input_files = glob(join(input_filepath, '*.npz'))
    #get learning signal contents
    ls_filepath = join(filePath, 'targets')
    ls_files = glob(join(ls_filepath, '*.npz'))
    
    #loop through inputs
    for fileInd in range(len(input_files)):
        # read in data from file
        this_input = load_data(input_files[fileInd])
        #concat
        for this_input_elem in this_input:
            if total_inputs is None:
                total_inputs = this_input_elem
            else:
                total_inputs = np.concatenate([total_inputs, this_input_elem],
                                        axis = 0)
        
        # get input number
        splits = input_files[fileInd].split('\\')
        if len(splits[-1]) == 11:
            input_num = splits[-1][-5]
        else:
            index_num = -(5 + (len(splits[-1]) - 11))
            input_num = splits[-1][index_num:-4]
            
        # find corresponding label
        match_string = f'target_{input_num}.npz'
        for file in ls_files:
            if match_string in file:
                this_ls = load_data(file)
                break
            
        #concat
        for this_ls_elem in this_ls:
            if total_learning_signals is None:
                total_learning_signals = this_ls_elem
            else:
                total_learning_signals = np.concatenate([total_learning_signals,
                                                   this_ls_elem],
                                                  axis = 0)
        
    # change into tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((total_inputs,
                                                  total_learning_signals))
    
    return dataset
        
                
def get_split_dataset(ds, indices):
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

#misc tools
def exp_convolve(tensor, decay):
    #filters a tensor by one time step
    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.get_shape()))
    transpose_perm = [1, 0] +list(r_shp)[2:]
    
    tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
    initializer = tf.zeros_like(tensor_time_major[0])
    filtered_tensor = tf.scan(lambda a, x: a * decay + (1-decay) * x, 
                              tensor_time_major, initializer = initializer)
    filtered_tensor = tf. transpose(filtered_tensor, perm=transpose_perm)
    return filtered_tensor

def shift_by_one_time_step(tensor, initializer = None):
    #shift input by one on the time dimension
    
    assert tensor.dtype in [tf.float16, tf.float32, tf.float64]
    r_shp = range(len(tensor.get_shape()))
    transpose_perm = [1,0] + list(r_shp)[2:]
    tensor_time_major = tf.transpose(tensor, perm=transpose_perm)
    
    if initializer is None:
        initializer = tf.zeros_like(tensor_time_major[0])
        
    shifted_tensor = tf.concat([initializer[None, :, :], tensor_time_major[:-1]], axis =0)
    shifted_tensor = tf.transpose(shifted_tensor, perm=transpose_perm)
    return shifted_tensor

#derivative and spike definition

def pseudo_derivative(v_scaled, dampening_factor):
    #define pseudo derivative
    return tf.maximum(1-tf.abs(v_scaled), 0) * dampening_factor

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    
    z_ = tf.greater(v_scaled, 0)
    z_ = tf.cast(z_, dtype = tf.float32)
    
    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled
        
        return [dE_dv_scaled,
                tf.zeros_like(dampening_factor)]
    return tf.identity(z_, name="SpikeFunction"), grad



#@tf.function
def LSNN_compute_eligibility_traces(decay_adaptation, beta, decay, dampening_factor,
                                    thr, v_scaled, z_pre, z_post, is_rec):
    n_neurons = tf.shape(z_post)[2]
    rho = decay_adaptation
    beta = beta
    alpha = decay
    
    #everything should be time major
    if len(tf.shape(z_pre)) == 5:
        z_pre_shape = tf.shape(z_pre)
        z_pre = tf.reshape(z_pre, shape = (z_pre_shape[0], z_pre_shape[1],
                                           z_pre_shape[-3]*z_pre_shape[-2]*z_pre_shape[-1]))
    z_pre = tf.transpose(z_pre, perm = [1,0,2])
    v_scaled = tf. transpose(v_scaled, perm = [1,0,2])
    z_post = tf.transpose(z_post, perm = [1,0,2])
    
    #compute psi
    psi = dampening_factor / thr * tf.maximum(tf.constant(0.,dtype = tf.float32),
                                                                  tf.constant(1.,dtype=tf.float32) - tf.abs(v_scaled))
    
    #update eligibility vector for memory
    update_epsilon_v = lambda epsilon_v, z_pre: alpha * epsilon_v + z_pre[:,:,None]
    epsilon_v_zero = tf.ones((1,1,n_neurons))* z_pre[0][:,:,None]
    epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer = epsilon_v_zero)
    epsilon_v = tf.concat([[epsilon_v_zero], epsilon_v], axis = 0)
    
    #update eligibility vector for adaptive threshold
    update_epsilon_a = lambda epsilon_a, elems:\
        (rho - beta * elems['psi'][:,None,:]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']
        
    epsilon_a_zero = tf.zeros_like(epsilon_v[0])
    epsilon_a = tf.scan(fn=update_epsilon_a,
                        elems = {'psi': psi[:-1], 'epsi': epsilon_v[:-1],
                                  'previous_epsi': shift_by_one_time_step(epsilon_v[:-1])},
                        initializer = epsilon_a_zero)
    
    epsilon_a = tf.concat([[epsilon_a_zero], epsilon_a], axis = 0)
    
    e_trace = psi[:,:,None,:] * (epsilon_v - beta * epsilon_a)
    
    #everything should be time major
    e_trace = tf.transpose(e_trace, perm = [1,0,2,3])
    epsilon_v = tf.transpose(epsilon_v, perm=[1,0,2,3])
    epsilon_a = tf.transpose(epsilon_a, perm=[1,0,2,3])
    psi = tf.transpose(psi, perm=[1,0,2])
    
    if is_rec:
        identity_diag = tf.eye(n_neurons)[None, None, :, :]
        e_trace -= identity_diag * e_trace
        epsilon_v -= identity_diag * epsilon_v
        epsilon_a -= identity_diag * epsilon_a
        
    return e_trace, epsilon_v, epsilon_a, psi

#@tf.function
def LSNN_compute_loss_gradient(params, learning_signal, z_pre, z_post,
                               v_post, b_post, decay_out = None, 
                               zero_on_diagonal = None,
                               experimental_relax_shapes = True):
    
    #unpack
    decay_adaptation = tf.exp(1/params.get('LN_decay_adaptation'))
    thr = params.get('LN_thr')
    decay = params.get('LN_decay')
    dampening_factor = params.get('LN_dampening_factor')
    beta = tf.cast(params.get('Beta_LN'), dtype = tf.float32)
    
    thr_post = thr + beta * b_post
    v_scaled = (v_post - thr_post) / thr
    
    e_trace, epsilon_v, epsilon_a, _ = LSNN_compute_eligibility_traces(
        decay_adaptation, beta, decay, dampening_factor, thr, v_scaled, 
        z_pre, z_post, zero_on_diagonal)
    
    if decay_out is not None:
        e_trace_time_major = tf.transpose(e_trace, perm=[1,0,2,3])
        filtered_e_zero = tf.zeros_like(e_trace_time_major[0])
        filtering = lambda filtered_e, e: filtered_e * decay_out + e * (1 - decay_out)
        filtered_e = tf.scan(filtering, e_trace_time_major, initializer = filtered_e_zero)
        filtered_e = tf.transpose(filtered_e, perm = [1,0,2,3])
        e_trace = filtered_e
        
    gradient = tf.einsum('btj, btij->ij', learning_signal, e_trace)
    return gradient, e_trace, epsilon_v, epsilon_a
        

#define custom classification loss
#@tf.function
def classification_loss(outputs, labels, class_weights, num_steps,
                        one_hot, parallel = False, experimental_relax_shapes = True):
    if one_hot:
        from tools_and_functions import softmax
    
        neuron_mem = outputs[1]
        if parallel:
            loss_fcn = tf.keras.losses.CategoricalCrossentropy(reduction =
                                                tf.keras.losses.Reduction.NONE)
        else:
            loss_fcn = tf.keras.losses.CategoricalCrossentropy()
            
        sample_weights = calculate_sample_weights(labels, class_weights,
                                                  one_hot)
        loss = tf.zeros(shape = [1,])
        for step in range(num_steps):
            loss_input = softmax(neuron_mem[:,step,:])
            loss += loss_fcn(labels, loss_input,
                            sample_weight = tf.reshape(sample_weights,
                                                             shape = [len(sample_weights),1]))
    
    else:
        loss_fcn = tf.keras.losses.BinaryCrossentropy()
        
        #calculate sample weights
        sample_weights = calculate_sample_weights(labels, class_weights)
        losses = loss_fcn(labels, outputs,
                          sample_weight = tf.reshape(sample_weights,
                                                     shape = [len(sample_weights),1]))
        loss = tf.reduce_sum(losses)
    
    return loss
    


#@tf.function
def get_metrics(output, labels, one_hot):
    if one_hot:
        #grab just the second set of indices
        output = tf.expand_dims(output[:,1], axis = 1)
        labels = tf.expand_dims(labels[:,1], axis = 1)
    pred_shape = tf.shape(output)
    label_shape = tf.shape(labels)
    if pred_shape[1] == 1 and len(label_shape) == 1:
        labels = tf.expand_dims(labels, axis = 1)
    labels = tf.cast(labels, dtype = tf.int64)
    is_correct = output == labels
    acc = tf.reduce_mean(tf.cast(is_correct,dtype=tf.float64))
    
    One_ind = tf.where(labels[:,0] == 1)
    Zero_ind = tf.where(labels[:,0] == 0)
    
    num_One_correct = sum(tf.gather(tf.cast(is_correct,dtype=tf.float64),
                                    indices = One_ind))
    num_Zero_correct = sum(tf.gather(tf.cast(is_correct,dtype=tf.float64),
                                    indices = Zero_ind))
    
    num_One_labels = sum(labels)
    num_Zero_labels = len(labels) - sum(labels)
    
    if num_One_labels > 0:
      tpr = tf.cast(num_One_correct[0,0],dtype=tf.int64)/num_One_labels
    else:
      tpr = tf.constant(np.zeros(shape = [1]), dtype = tf.float64)
    if num_Zero_labels > 0:
      tnr = tf.cast(num_Zero_correct[0,0],dtype=tf.int64)/num_Zero_labels
    else:
      tnr = tf.constant(np.zeros(shape = [1]), dtype = tf.float64)
    
    acc = tf.expand_dims(acc, axis = 0)
    
    metrics = tf.concat([acc,tpr,tnr], axis = 0)
    return metrics

@tf.function
def test_classification_acc(spikes, labels, output_weights,
                            experimental_relax_shapes = True):
    spike_count = tf.cast(tf.reduce_sum(spikes,1),dtype= tf.float64)
    
    logits = spike_count @ output_weights
    
    is_correct = tf.argmax(logits, axis = -1) == labels
    num_correct = tf.math.reduce_sum(tf.cast(is_correct,dtype=tf.int64))
    return num_correct

#@tf.function
def SGD(weights, grad_list, learning_rate):
    
    weights_new = [0, 0]
    for grad_ind in range(len(grad_list)):
        grad = grad_list[grad_ind]
        weights_new[grad_ind] = tf.math.subtract(weights[grad_ind],
                                             tf.math.multiply(learning_rate, grad))

    return weights_new


def softmax(inputs):
    #this function assumes the inputs are structured as such
    #batch, d1    
    for in_vec in range(tf.shape(inputs)[0]):
        in_slice = inputs[in_vec,:]
        softmax_slice = tf.math.exp(in_slice) / tf.math.reduce_sum(tf.math.exp(in_slice))
        #store
        if in_vec == 0:
            out = tf.expand_dims(softmax_slice, axis = 0)
        else:
            out = tf.concat([out, tf.expand_dims(softmax_slice, axis = 0)], 
                            axis = 0)
        
    return out


#load dictionary from json
def load_dict(path_to_dict):
    from json import loads
    
    #read in
    with open(path_to_dict, 'r') as f:
        myDict = f.read()
        
    #load
    dict_to_return = loads(myDict)
    
    return dict_to_return


#save dictionary to json
def save_dict(save_path, dict_to_save):
    from json import dumps
    
    with open(save_path, 'w') as convert_file:
        convert_file.write(dumps(dict_to_save))
        
#load weights
def load_weights(layer, load_from_path):
    from numpy import load
    
    ind = 0
    com = "layer.set_weights(["
    weights = load(load_from_path)
    for w in weights:
        ind += 1
        if ind == 1:
            com += f"weights['{w}']"
        elif ind == len(weights):
            com += f", weights['{w}']])"
        else:
            com += f", weights['{w}']"
    exec(com)
    
    return layer

#save weights
def save_weights(weights, save_to_path):
    weights_list = []
    loopInd = -1
    com = "np.savez(save_to_path"
    for this_weight in weights:
        loopInd += 1
        weights_list.append(this_weight)
        com_add = f", weights_list[{loopInd}]"
        com += com_add
    com += ")"
    exec(com)
    
#save data
def save_data(data, save_to_path):
    loopInd = -1
    com = "np.savez(save_to_path"
    for data_point in data:
        loopInd += 1
        com_add = f", data[{loopInd}]"
        com += com_add
    com += ")"
    exec(com)
    
def load_data(load_from_path):
    from numpy import load
    
    data = []
    this_data = load(load_from_path)
    for d in this_data:
        data.append(this_data[f'{d}'])

    return data
    
#initialize optimizer
def init_optimizer(optimizer, opt_var_path, layers, update_var):
    from os.path import isfile
    
    #create dummy grads
    dummy_grads = []
    for key in layers:
        layer = layers[key]
        layer_var = layer.trainable_variables
        for var in layer_var:
            dummy_grads.append(tf.zeros_like(var))
    
    #load optimizer weights if they exist
    if isfile(opt_var_path):   
        #apply dummy gradients to initialize optimizer
        optimizer.apply_gradients([(g,v) for g,v in zip(dummy_grads,update_var)])
        #set optimizer variables
        optimizer = load_weights(optimizer, opt_var_path)
        
    return None

#calculate class weights
def calculate_class_weights(dataset, one_hot):
    from numpy import array
    
    y_targets = array([target.numpy() for _, target in iter(dataset)],
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


#calculate sample weights
def calculate_sample_weights(labels, class_weights, one_hot):
    from numpy import zeros
    
    sample_weights = zeros(shape = [tf.shape(labels)[0],1])
    count = -1
    for label in labels:
        count += 1
        if one_hot:
            if label[1] == 1:
                sample_weights[count] = class_weights[1]
            else:
                sample_weights[count] = class_weights[0]
        else:
            if label == 1:
                sample_weights[count] = class_weights[1]
            else:
                sample_weights[count] = class_weights[0]
    sample_weights = tf.convert_to_tensor(sample_weights)
    
    return sample_weights
        
        

## plotting functions
def plot_train_metrics(metric_hist, savefigfile):
    #input: a list of driver metrics lists by epoch where each driver metric list
    #contains a list of lists of each driver's specific metrics
    #epochs -> drivers -> individual driver -> individual metric
    
    #get average final batch metrics across drivers vs outer loop epochs
    ave_final_metrics_by_epoch = np.zeros([3,len(metric_hist)])
    ep_ind = -1
    for driver_list in metric_hist:
        ep_ind += 1
        #should pull out the list of driver's list of metrics for each epoch
        sum_across_drivers_for_epoch = 0
        num_drivers = 0
        for driver in driver_list:
            #should pull out individual driver's list of metrics
            #grab final metric
            sum_across_drivers_for_epoch += driver[-1]
            num_drivers += 1
        #divide by number of drivers
        ave_across_drivers_for_epoch = sum_across_drivers_for_epoch/num_drivers
        #store
        ave_final_metrics_by_epoch[:,ep_ind] = tf.reshape(ave_across_drivers_for_epoch, shape = [3])
        
    #create plots using matplotlib
    num_epochs = np.arange(1,len(metric_hist)+1)
    plt.plot(num_epochs, ave_final_metrics_by_epoch[0,:],label = 'Accuracy')
    plt.plot(num_epochs, ave_final_metrics_by_epoch[1,:],label = 'TPR')
    plt.plot(num_epochs, ave_final_metrics_by_epoch[2,:],label = 'TNR')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Values')
    plt.legend()
    plt.title('Average Final Batch Metrics across Drivers per Epoch')
    plt.savefig(savefigfile + '/AveMetricsPerEpoch.png')
    plt.close()
    
def plot_outer_loop_loss(outer_loop_loss_hist, savefigfile):
    
    num_updates = np.arange(1,len(outer_loop_loss_hist)+1)
    plt.plot(num_updates,outer_loop_loss_hist)
    plt.xlabel('Outer Loop Updates')
    plt.ylabel('Loss Value')
    plt.title('Outer Loop Loss per Update')
    plt.savefig(join(savefigfile, 'OuterLoopLossPerUpdate.png'))
    plt.close()
    
def plot_test_epochs_per_driver(metrics_hist_per_driver, savefigfile, batch_size):
    #create subplots equal to the number of drivers
    fig, axs = plt.subplots(len(metrics_hist_per_driver), figsize = (10,6))
    fig.suptitle('Driver Metrics per Epoch Trained with E-prop')
    driver_ind = -1
    for metrics in metrics_hist_per_driver:
        driver_ind += 1
        axs[driver_ind].plot(np.arange(1,len(metrics[0,:])+1), metrics[0,:],
                             label = 'Acc', marker = 'o')
        axs[driver_ind].plot(np.arange(1,len(metrics[1,:])+1), metrics[1,:],
                             label = 'TPR', marker = 'o')
        axs[driver_ind].plot(np.arange(1,len(metrics[2,:])+1), metrics[2,:], 
                             label = 'TNR', marker = 'o')
        axs[driver_ind].set_title(f'Driver {driver_ind + 1}')
        axs[driver_ind].set(xlabel = 'Number of Epochs', ylabel = 'Percent')
        axs[driver_ind].legend()

    fig.tight_layout()
    plt.savefig(join(savefigfile,f'MetricsvsEpochs_BatchSize_{batch_size}.png'))
    plt.close(fig)
    