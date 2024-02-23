# -*- coding: utf-8 -*-
"""
Created on Sun May 21 10:54:13 2023

@author: nalmrb
"""

def meta_learn_train(data_paths, paths, params):
    
    #import
    import tensorflow as tf
    import numpy as np
    import os
    from tools_and_functions import load_dict, save_dict
    from tools_and_functions import calculate_class_weights, calculate_sample_weights
    from tools_and_functions import init_optimizer, SGD, get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import Load_driver_data_random_subset
    from tools_and_functions import save_weights, classification_loss
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import LSNN_func, create_LSG_model
    from Models_and_Layers import conv_forward_pass, get_output
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    #storage flags
    store_metrics = 1
    store_loss = 1
    
    #define paths
    train_config = params.get('train_config')
    store_path = paths.get('store_path')
    if train_config == 0:
        load_weights_path = ''
    else:
        load_weights_path = paths.get('weights_path')
        
    #define LN_params, MetaL_params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    
    #extract parameters
    window_size = MetaL_params.get('window_size')
    learning_rate_inner = MetaL_params.get('learning_rate_inner')
    learning_rate_outer = MetaL_params.get('learning_rate_outer')
    num_steps = LN_params.get('num_steps')
    epochs = MetaL_params.get('epochs_train')
    num_data_per_driver = MetaL_params.get('num_data_per_driver')
    num_updates = MetaL_params.get('num_updates')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
        
    #define counters
    counters = load_dict(os.path.join(store_path, 'counters.txt'))
    saveMetricNum = counters.get('saveMetricNum')
    saveOutLoopLossNum = counters.get('saveOutLoopLossNum')
    
    #define models, loss, outer loop trainable variables and ini optimizer
    with strategy.scope():
        #define conv layer
        conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                             'conv_kernel_size':LN_params.get('conv_kernel_size'),
                             'window_size':window_size}
        
        conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params,
                                                            load_weights_path = load_weights_path)
        
        #create dummy input for LN init
        dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
        dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                              dummy_input, num_steps, training = True)
        
        #define LN model
        LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                                                 load_weights_path = load_weights_path,
                                                                                 dummy_in = dummy_conv_out,
                                                                                 output_version = output_version)
        
        #extract weights
        LN_Hidden_weights = LN_Hidden_layer.trainable_variables
        
        #define LSG model
        #update meta l params with num steps
        MetaL_params.update({'num_steps':num_steps})
        
        #dummy input
        dummy_LSG_input_spike_shape = tf.shape(dummy_conv_out)
        dummy_LSG_input_spike = tf.reshape(dummy_conv_out,
                                           shape = (dummy_LSG_input_spike_shape[0],
                                                    dummy_LSG_input_spike_shape [1],
                                                    dummy_LSG_input_spike_shape[-3]*
                                                    dummy_LSG_input_spike_shape[-2]*
                                                    dummy_LSG_input_spike_shape[-1]))
        dummy_LSG_inputs = tf.concat([dummy_LSG_input_spike,
                                                 dummy_LN_Hidden_outs[0]], axis = 2)
        
        #build model
        Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                                        load_weights_path = load_weights_path,
                                                        dummy_in = dummy_LSG_inputs)
        
        #define loss function within distributed scope
        if output_version == 2:
            def compute_loss(outputs, labels, class_weights, num_steps,
                             one_hot, global_batch_size):
                loss = classification_loss(outputs, labels, class_weights,
                                               num_steps, one_hot,
                                               parallel = True)
                
                #scale loss
                scale_loss = tf.math.divide(tf.reduce_sum(loss), 
                                            tf.cast(global_batch_size,
                                                    dtype = tf.float32))
                return scale_loss
                                      
        else:      
            def compute_loss(outputs, labels, class_weights, global_batch_size):
                out_sigmoid = outputs[0]
                loss_fcn = tf.keras.losses.BinaryCrossentropy(reduction = 
                                                              tf.keras.losses.Reduction.NONE)
                
                #calculate sample weights
                sample_weights = calculate_sample_weights(labels, class_weights,
                                                          one_hot)

                losses = loss_fcn(labels, out_sigmoid,
                                  sample_weight = tf.reshape(sample_weights,
                                                             shape = [len(sample_weights),1]))
                scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                            tf.cast(global_batch_size,dtype=tf.float32))
                
                return scale_loss
        
        #define variables for outer loop optimization
        conv_layer_1 = conv_layers_dict.get('layer_1')
        conv_layer_2 = conv_layers_dict.get('layer_2')
        
        var_BPTT = conv_layer_1.trainable_variables +\
            conv_layer_2.trainable_variables +\
                Hidden_LSG.trainable_variables + LSG_out.trainable_variables +\
                    LN_output_layer.trainable_variables
                    
        #define and init optimizer
        optimizer_outer = tf.keras.optimizers.Adam(learning_rate = learning_rate_outer)
        
        #opt var path
        opt_var_path = store_path + 'opt_var.npz'
        
        #define set of layers that will be optimized
        layers = {'conv_layer_1':conv_layer_1, 'conv_layer_2':conv_layer_2,
                  'Hidden_LSG':Hidden_LSG, 'LSG_out':LSG_out, 
                  'LN_output':LN_output_layer, 'LN_Hidden':LN_Hidden_layer}
        
        #init
        strategy.run(init_optimizer, args = (optimizer_outer, opt_var_path,
                                             layers, var_BPTT + LN_Hidden_weights))
    
    #define train function
    def train_func(inputs, targets):
        
        with tf.GradientTape() as tape:
            tape.watch(LN_W_Surro)
            
            #forward pass
            conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                         inputs, num_steps, training = True)
            
            LN_Hidden_out = LSNN_func(conv_out, LN_W_Surro, LN_params)
            outputs_1 = get_output(LN_Hidden_out[0], LN_output_layer,
                                   ol_training = True)
            
            #extract predictions
            if LN_output_layer.name == "LN_output_layer_v2":
                out_pred_1 = outputs_1[2]
            else:
                out_pred_1 = outputs_1[1]
                
            conv_out_shape = tf.shape(conv_out)
            LSG_input_spike = tf.reshape(conv_out,
                                         shape = (conv_out_shape[0],
                                                  conv_out_shape[1],
                                                  conv_out_shape[-3]*
                                                  conv_out_shape[-2]*
                                                  conv_out_shape[-1]))
            LSG_inputs = tf.concat([LSG_input_spike,
                                    LN_Hidden_out[0]], axis = 2)
            Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
            Learning_signal = get_output(Hidden_LSG_outs[0], LSG_out,
                                         ol_training = True)
            
            #compute eprop gradients            
            [grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                LN_params,
                Learning_signal,
                conv_out,LN_Hidden_out[0],
                LN_Hidden_out[1],LN_Hidden_out[2],
                zero_on_diagonal = False, decay_out = None)
           
            #grad for recurrent
            z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                         LN_Hidden_out[0][:,:-1]], axis=1)
            [grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                LN_params, Learning_signal, z_previous_step,
                LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
                decay_out = None)   
                
            #concatenate gradients
            grad_list = [grad_in, 
                         grad_rec]
            
            #apply gradients using SGD
            LN_W_Surro_new = SGD(LN_W_Surro, grad_list, learning_rate_inner)
            
            #second pass
            LN_Hidden_outs = LSNN_func(conv_out, LN_W_Surro_new, LN_params)
            outputs_2 = get_output(LN_Hidden_outs[0], LN_output_layer,
                                   ol_training = True)
            
            #extract predictions
            if LN_output_layer.name == "LN_output_layer_v2":
                out_pred_2 = outputs_2[2]
            else:
                out_pred_2 = outputs_2[1]
            
            #compute loss
            batch_size = tf.shape(targets)[0]
            loss = compute_loss(outputs_2, targets, class_weights, num_steps,
                                one_hot, batch_size)
            
        #calculate outer loop gradients
        grads_BPTT = tape.gradient(loss, var_BPTT + LN_W_Surro)
        
        #apply gradients
        optimizer_outer.apply_gradients([(g,v) for g,v in zip(grads_BPTT, 
                                                              var_BPTT + LN_Hidden_weights)])
        
        return loss, out_pred_1, out_pred_2, targets
    
    #define the distributed wrapped
    def dist_train_step(inputs, targets):
        
        per_replica_losses, per_replica_outputs_pre, per_replica_outputs_post, per_replica_targets = strategy.run(train_func,
                args = (inputs, targets))
        
        if strategy.num_replicas_in_sync > 1:
            total_outputs_pre = tf.concat(per_replica_outputs_pre.values, axis = 0)
            total_outputs_post = tf.concat(per_replica_outputs_post.values, axis = 0)
            total_targets = tf.concat(per_replica_targets.values, axis = 0)
        else:
            total_outputs_pre = per_replica_outputs_pre
            total_outputs_post = per_replica_outputs_post
            total_targets = per_replica_targets
            
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis = None), total_targets, total_outputs_pre, total_outputs_post
            
    #get number of drivers in train folder
    numTrainDrivers = len(data_paths)
    
    #initialize temporary metric storage
    if store_metrics:
        metrics_post = np.zeros([3,numTrainDrivers,epochs])
        metrics_pre = np.zeros([3,numTrainDrivers,epochs])
        
    if store_loss:
        loss_data = np.zeros([epochs,])
    
    #training loop
    for epoch in range(epochs):
        
        #initialize epoch_loss
        if store_loss:
            epoch_loss = 0
            
        #create conglomerate dataset using data from each driver
        for driver_index in range(numTrainDrivers):
            if driver_index == 0:
                #select driver dataset
                train_dataset, num_trials_total = Load_driver_data_random_subset(data_paths[driver_index], num_data_per_driver,
                                                                                     one_hot = one_hot)
            else:
                #select driver dataset
                driver_dataset, num_trials = Load_driver_data_random_subset(data_paths[driver_index], num_data_per_driver,
                                                                                     one_hot = one_hot)
                #append to train_dataset
                train_dataset = train_dataset.concatenate(driver_dataset)
                num_trials_total = num_trials_total + num_trials
                
        batch_size = tf.cast(num_trials_total/num_updates, dtype = tf.int64)
            
        #batch
        train_dataset = train_dataset.batch(batch_size)
            
        #distribute
        train_dataset_dist = strategy.experimental_distribute_dataset(train_dataset)
            
        #calculate class weight from dataset
        class_weights = calculate_class_weights(train_dataset, one_hot)
        
        #train
        for inputs, targets in train_dataset_dist:
            #create surrogate weights
            LN_W_Surro = [tf.identity(LN_Hidden_weights[0]),
                   tf.identity(LN_Hidden_weights[1])]
            
            loss, targets, outputs_pre, outputs_post = dist_train_step(inputs, targets)
            
            #increment epoch loss
            epoch_loss = epoch_loss + loss.numpy()
            
            
        #store metrics
        if store_metrics:
            for driver_index in range(numTrainDrivers):
                metrics_pre[:,driver_index,epoch] = get_metrics(outputs_pre[(0 + num_data_per_driver*driver_index):(num_data_per_driver-1 + num_data_per_driver * driver_index)], 
                                          targets[(0 + num_data_per_driver*driver_index):(num_data_per_driver-1 + num_data_per_driver * driver_index)],
                                          one_hot = one_hot)
                metrics_post[:,driver_index,epoch] = get_metrics(outputs_post[(0 + num_data_per_driver*driver_index):(num_data_per_driver-1 + num_data_per_driver * driver_index)],
                                       targets[(0 + num_data_per_driver*driver_index):(num_data_per_driver-1 + num_data_per_driver * driver_index)],
                                       one_hot = one_hot)
            
        #save outer_loop_loss_value
        if store_loss:
            loss_data[epoch] = epoch_loss
                
        #save weights every 10 epochs        
        if (epoch % 10) == 0: 
            save_weights(conv_layer_1.get_weights(),
                         save_to_path = os.path.join(load_weights_path,
                                                     'LN_conv_layer_1_weights.npz'))
            save_weights(conv_layer_2.get_weights(), 
                         save_to_path = os.path.join(load_weights_path,
                                                     'LN_conv_layer_2_weights.npz'))
            save_weights(LN_Hidden_weights, 
                         save_to_path = os.path.join(load_weights_path,
                                                     'LN_Hidden_weights.npz'))
            save_weights(LN_output_layer.get_weights(), 
                         save_to_path = os.path.join(load_weights_path,
                                                     'LN_output_weights.npz'))
            save_weights(Hidden_LSG.get_weights(),
                         save_to_path = os.path.join(load_weights_path,
                                                     'Hidden_LSG_weights.npz'))
            save_weights(LSG_out.get_weights(),
                         save_to_path = os.path.join(load_weights_path,
                                                     'LSG_out_weights.npz'))
            
    
    #save metrics and loss
    if store_metrics:
        np.save(os.path.join(store_path, 'training_metrics_pre'), metrics_pre)
        np.save(os.path.join(store_path, 'training_metrics_post'),metrics_post)
    if store_loss:
        np.save(os.path.join(store_path, 'training_loss'),loss_data)
        
    #save final weights
    save_weights(conv_layer_1.get_weights(),
                 save_to_path = os.path.join(load_weights_path,
                                             'LN_conv_layer_1_weights.npz'))
    save_weights(conv_layer_2.get_weights(), 
                 save_to_path = os.path.join(load_weights_path,
                                             'LN_conv_layer_2_weights.npz'))
    save_weights(LN_Hidden_weights, 
                 save_to_path = os.path.join(load_weights_path,
                                             'LN_Hidden_weights.npz'))
    save_weights(LN_output_layer.get_weights(), 
                 save_to_path = os.path.join(load_weights_path,
                                             'LN_output_weights.npz'))
    save_weights(Hidden_LSG.get_weights(),
                 save_to_path = os.path.join(load_weights_path,
                                             'Hidden_LSG_weights.npz'))
    save_weights(LSG_out.get_weights(),
                 save_to_path = os.path.join(load_weights_path,
                                             'LSG_out_weights.npz'))
    
    #save optimizer variables
    save_weights(optimizer_outer.variables(), 
                 save_to_path = os.path.join(store_path, 'opt_var.npz'))
    
    #save counters
    counters = {'saveMetricNum':saveMetricNum, 'saveOutLoopLossNum':saveOutLoopLossNum}
    save_dict(os.path.join(store_path, 'counters.txt'), counters)
    
    
    
def meta_learn_test(data_paths, paths, params):
    #define imports
    import tensorflow as tf
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tools_and_functions import get_split_dataset
    from tools_and_functions import SGD, get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import plot_test_epochs_per_driver
    from tools_and_functions import load_driver_data_in_order_window
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import create_LSG_model, LSNN_func
    from Models_and_Layers import conv_forward_pass, get_output
    
    #define paths
    store_path = paths.get('store_path')
    load_weights_path = paths.get('weights_path')
    saveFigFilePath = paths.get('results_path')
        
    #define LN_params, MetaL_params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    
    #extract parameters
    window_size = MetaL_params.get('window_size')
    learning_rate_inner = MetaL_params.get('learning_rate_inner')
    learning_rate_outer = MetaL_params.get('learning_rate_outer')
    num_steps = LN_params.get('num_steps')
    epochs = MetaL_params.get('epochs_test')
    num_updates = MetaL_params.get('num_updates')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
    
    # loop to try different batch sizes and document the results in plot format
    batch_sizes = [5, 15, 30, 50, 100, 200, 9999]
    for batch_size in batch_sizes:
        
        #load models
        #define conv layer
        conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                             'conv_kernel_size':LN_params.get('conv_kernel_size'),
                             'window_size':window_size}
        
        conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params,
                                                            load_weights_path = load_weights_path)
        
        #create dummy input for LN init
        dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
        dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                              dummy_input, num_steps, training = True)
        
        #define LN model
        LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                                                 load_weights_path = load_weights_path,
                                                                                 dummy_in = dummy_conv_out,
                                                                                 output_version = output_version)
        
        #extract weights
        LN_Hidden_weights = LN_Hidden_layer.trainable_variables
        
        #define LSG model
        #update meta l params with num steps
        MetaL_params.update({'num_steps':num_steps})
        
        #dummy input
        dummy_LSG_input_spike_shape = tf.shape(dummy_conv_out)
        dummy_LSG_input_spike = tf.reshape(dummy_conv_out,
                                           shape = (dummy_LSG_input_spike_shape[0],
                                                    dummy_LSG_input_spike_shape [1],
                                                    dummy_LSG_input_spike_shape[-3]*
                                                    dummy_LSG_input_spike_shape[-2]*
                                                    dummy_LSG_input_spike_shape[-1]))
        dummy_LSG_inputs = tf.concat([dummy_LSG_input_spike,
                                                 dummy_LN_Hidden_outs[0]], axis = 2)
        
        #build model
        Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                                        load_weights_path = load_weights_path,
                                                        dummy_in = dummy_LSG_inputs)
        
        #get number of drivers
        numTestDrivers = len(data_paths)
        
        #validation loop
        metrics_hist_per_driver = []
        for driver_index in range(numTestDrivers):
            
            #create dataset
            driver_dataset, num_trials = load_driver_data_in_order_window(data_paths[driver_index], window_size, one_hot = one_hot)
            if window_size == 1:
                train_split_frac = 0.8
            elif window_size == 2:
                train_split_frac = 0.75
            elif window_size == 3:
                train_split_frac = 0.66
            else:
                train_split_frac = 0.8        
            if batch_size == 9999:
                batch_size_train = tf.cast(np.round(num_trials*train_split_frac)/num_updates,
                                           dtype = tf.int64)
            else:
                batch_size_train = batch_size
            batch_size_test = tf.cast(np.round(num_trials*(1-train_split_frac))/num_updates,
                                      dtype = tf.int64)
            
            #split into training and testing sets
            #get labels
            y_targets = np.array([ target.numpy() for _, target in iter(driver_dataset) ],
                                 dtype = object)
            x_indices = np.arange(len(y_targets))
            
            #get split indices
            x_tr_ind, x_te_ind, y_tr_ind, y_te_ind = train_test_split(x_indices,
                                                                      y_targets,
                                                                      test_size=(1-train_split_frac), 
                                                                      stratify=y_targets, 
                                                                      random_state=53)
            train_dataset = get_split_dataset(driver_dataset, x_tr_ind)
            train_dataset = train_dataset.batch(batch_size_train)
            
            test_dataset = get_split_dataset(driver_dataset, x_te_ind)
            test_dataset = test_dataset.batch(batch_size_test)
            
            #copy LN weights to tf variable
            LN_W_Surro = [tf.identity(LN_Hidden_weights[0]),
              tf.identity(LN_Hidden_weights[1])]
            
            metrics_hist = np.zeros([3,epochs])
            for ep_cnt in range(epochs):
                if (ep_cnt % 5) == 0:
                    print(f"Testing Epoch {ep_cnt}")
                if ep_cnt > 0:
                    for step, (inputs, targets) in enumerate(train_dataset):
                        
                        #forward pass
                        conv_out = conv_forward_pass(conv_layers_dict, 
                                                     LIF_layers_dict, inputs,
                                                     num_steps)
                        
                        LN_Hidden_out = LSNN_func(conv_out, LN_W_Surro, 
                                                  LN_params)
                        
                        LSG_input_spike_shape = tf.shape(conv_out)
                        LSG_input_spike = tf.reshape(conv_out,
                                                           shape = (LSG_input_spike_shape[0],
                                                                    LSG_input_spike_shape [1],
                                                                    LSG_input_spike_shape[-3]*
                                                                    LSG_input_spike_shape[-2]*
                                                                    LSG_input_spike_shape[-1]))
                        LSG_inputs = tf.concat([LSG_input_spike,
                                                        LN_Hidden_out[0]], axis = 2)
                        Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
                        Learning_signal = get_output(Hidden_LSG_outs[0], LSG_out, ol_training = False)
                        Learning_signal_shape = tf.cast(Learning_signal, dtype = tf.float32)
                        
                        #eprop
                        [grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                            LN_params,
                            Learning_signal,
                            conv_out,LN_Hidden_out[0],
                            LN_Hidden_out[1],LN_Hidden_out[2],
                            zero_on_diagonal = False, decay_out = None)
                       
                        #grad for recurrent
                        z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                                     LN_Hidden_out[0][:,:-1]], axis=1)
                        [grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                            LN_params, Learning_signal, z_previous_step,
                            LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2],
                            zero_on_diagonal = True,
                            decay_out = None)
                        
                        #concatenate gradients
                        grad_list = [grad_in, 
                                   grad_rec]
                        
                        #apply gradients via SGD
                        LN_W_Surro = SGD(LN_W_Surro, grad_list,
                                         learning_rate_inner)
        
                #test
                for step, (inputs, targets) in enumerate(test_dataset):
                    if "total_targets" in locals():
                        total_targets = tf.concat([total_targets,targets], 
                                                  axis = 0)
                    else:
                        total_targets = targets
                        
                    #forward pass
                    conv_out = conv_forward_pass(conv_layers_dict,
                                                    LIF_layers_dict, inputs,
                                                    num_steps)
                    
                    LN_Hidden_out = LSNN_func(conv_out, LN_W_Surro, LN_params)
                    out = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = False)
                    
                    #extract outputs
                    if LN_output_layer.name == "LN_output_layer_v2":
                        out_pred = out[2]
                    else:
                        out_pred = out[1]
                    
                    #save output
                    if "total_outputs" in locals():
                        total_outputs = tf.concat([total_outputs,
                                                   out_pred.numpy()],
                                                  axis = 0)
                    else:
                         total_outputs = out_pred.numpy()
                
                #get metrics
                te_metrics = get_metrics(total_outputs, total_targets,
                                         one_hot = one_hot)
                te_acc = te_metrics[0]
                te_TPR = te_metrics[1]
                if ep_cnt == 0 or (ep_cnt+1 % 5) == 0:
                   print(f"Accuracy: {te_metrics[0]}")
                   print(f"TPR: {te_metrics[1]}")
                   print(f"TNR: {te_metrics[2]}")
                   
                #store metrics for plotting
                metrics_hist[:, ep_cnt] = te_metrics.numpy()
                
                del total_targets
                del total_outputs
            
            #store metrics for plotting
            metrics_hist_per_driver.append(metrics_hist)
            
        #make plots
        plot_test_epochs_per_driver(metrics_hist_per_driver, saveFigFilePath,
                                    batch_size)
        
        
def Pretrain_LN(paths, params):
    
    import tensorflow as tf
    import numpy as np
    import os
    from sklearn.model_selection import train_test_split
    from tools_and_functions import get_split_dataset
    from tools_and_functions import Load_dataset
    from tools_and_functions import calculate_class_weights, save_weights
    from tools_and_functions import classification_loss, get_metrics
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import conv_forward_pass, get_output
    
    #unpack paths
    save_final_weights_path = paths.get("LN_Pretrain_Final_weights_path")
    data_path = paths.get('exp1_driver_data')
    
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    pretrain_params = params.get('pretrain_params')
    
    debug = pretrain_params.get('debug')
    if debug is None:
        debug = False
    batch_size = pretrain_params.get('batch_size')
    learning_rate = pretrain_params.get('learning_rate')
    max_epochs = pretrain_params.get('epochs_pretrain')
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    window_size = MetaL_params.get('window_size')
    num_steps = LN_params.get('num_steps')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
    
    #load models
    #define conv layer
    conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                             'conv_kernel_size':LN_params.get('conv_kernel_size'),
                             'window_size':window_size}
    
    conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params)
    
    #create dummy input for LN init
    dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
    dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                          dummy_input, num_steps, training = True)
    
    #define LN model
    LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                    dummy_in = dummy_conv_out,
                                                    output_version = output_version)
    
    #read in experiment 1 data
    exp1_dataset, exp1_dataset_length = Load_dataset(data_path, one_hot = one_hot,
                                                     debug = debug)
    
    #calculate class weight from dataset
    class_weights = calculate_class_weights(exp1_dataset, one_hot)
    
    #create train test split
    #get labels
    y_targets = np.array([target.numpy() for _, target in iter(exp1_dataset)],
                         dtype = object)
    x_indices = np.arange(len(y_targets))
    
    #get split indices
    if window_size == 1:
        train_split_frac = 0.8
    elif window_size == 2:
        train_split_frac = 0.75
    elif window_size == 3:
        train_split_frac = 0.66
    else:
        train_split_frac = 0.8 
    x_tr_ind, x_te_ind, y_tr_ind, y_te_ind = train_test_split(x_indices,
                                                              y_targets,
                                                              test_size = (1-train_split_frac),
                                                              stratify = y_targets,
                                                              random_state = 42)
    
    train_dataset = get_split_dataset(exp1_dataset, x_tr_ind)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    
    test_dataset = get_split_dataset(exp1_dataset, x_te_ind)
    test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    
    #define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
    #specify trainable variables
    conv_layer_1 = conv_layers_dict.get('layer_1')
    conv_layer_2 = conv_layers_dict.get('layer_2')
    train_var = conv_layer_1.trainable_variables +\
        conv_layer_2.trainable_variables +\
            LN_Hidden_layer.trainable_variables +\
            LN_output_layer.trainable_variables
    
    #main loop
    record_weights = 0
    for epoch in range(max_epochs):
        epoch_loss = 0
        for step, (inputs, targets) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                [tape.watch(v) for v in train_var]
                
                #forward pass
                conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                             inputs, num_steps, training = True)
                LN_Hidden_out = LN_Hidden_layer(conv_out)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                
                #compute loss
                loss = classification_loss(outputs, targets, class_weights,
                                               num_steps, one_hot)
                
                #add to epoch loss
                epoch_loss += loss.numpy()
                
            #compute gradients
            grads = tape.gradient(loss, train_var)
            
            #apply gradients
            optimizer.apply_gradients([(g,v) for g,v in zip(grads,train_var)])
            
        
        #test
        for step, (inputs, targets) in enumerate(test_dataset):
                
            #forward pass
            conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                         inputs, num_steps, training = True)
            LN_Hidden_out = LN_Hidden_layer(conv_out)
            outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
            #extract outputs
            if LN_output_layer.name == "LN_output_layer_v2":
                out_pred = outputs[2]
            else:
                out_pred = outputs[1]
            
            #save outputs and targets
            if step == 0:
                total_targets = targets
                total_outputs = out_pred.numpy()
            else:
                total_outputs = tf.concat([total_outputs, out_pred.numpy()],axis = 0)
                total_targets = tf.concat([total_targets,targets],axis = 0)                
                 
        #get metrics
        metrics = get_metrics(total_outputs, total_targets, one_hot = one_hot)
        
        #print achieved metrics
        print(f"Achieved Testing Accuracy: {metrics[0]}")
        print(f"Achieved Testing TPR:  {metrics[1]}")
        print(f"Achieved Testing TNR: {metrics[2]}")
        
        #check if satisfactory performance is met
        if metrics[0] >= 0.90 and metrics[1] >= 0.90 and metrics[2] >= 0.95:
            record_weights = 1
            break
            
    #save weights if performance metrics were met
    if record_weights == 1:
        succeeded = 1
        print("performance criteria met, saving weights")
        
        if not os.path.exists(save_final_weights_path):
            os.mkdir(save_final_weights_path)
        
        save_weights(conv_layer_1.get_weights(), 
                     os.path.join(save_final_weights_path,
                                  'LN_conv_layer_1_weights.npz'))
        save_weights(conv_layer_2.get_weights(),
                     os.path.join(save_final_weights_path,
                                  'LN_conv_layer_2_weights.npz'))
        save_weights(LN_Hidden_layer.get_weights(),
                     os.path.join(save_final_weights_path,
                                  'LN_Hidden_weights.npz'))
        save_weights(LN_output_layer.get_weights(), 
                     os.path.join(save_final_weights_path,
                                  'LN_output_weights.npz'))
    else:
        succeeded = 0 
        print("performance criteria not met.")
        
    return succeeded


def LN_supervised_learning_with_eprop(paths, params):
    
    import tensorflow as tf
    import numpy as np
    import os
    import shutil
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from tools_and_functions import get_split_dataset
    from tools_and_functions import Load_dataset
    from tools_and_functions import calculate_class_weights
    from tools_and_functions import classification_loss, get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import save_weights, save_data
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import conv_forward_pass, get_output
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    SL_params = params.get('SL_params')
    transfer_learning = SL_params.get('transfer_learning')
    if transfer_learning is None:
        transfer_learning = 0
    elif transfer_learning==1:
        custom_weights_path = SL_params.get('custom_weights_path')
    batch_size = SL_params.get('batch_size')
    learning_rate = SL_params.get('learning_rate')
    max_epochs = SL_params.get('epochs_train')
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    window_size = MetaL_params.get('window_size')
    num_steps = LN_params.get('num_steps')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
    
    #unpack paths
    store_path = paths.get('LN_LSG_supervised_learning_store_path')
    save_weights_path = os.path.join(store_path, 'weights')
    save_results_path = os.path.join(store_path, 'results')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    if os.path.exists(save_weights_path):
        if transfer_learning != 1:
            shutil.rmtree(save_weights_path)
            os.mkdir(save_weights_path)
            load_weights_path = ''
            data_path = paths.get('exp1_driver_data')
        else:
            if custom_weights_path is None:
                load_weights_path = save_weights_path
            else:
                load_weights_path = custom_weights_path
            data_path = paths.get('train_drivers_data')
    else:
        if transfer_learning == 1:
            # incorrect config
            print("Previous weights do not exist, can't do transfer learning")
            import sys
            sys.exit()
        else:
            os.mkdir(save_weights_path)
            load_weights_path = ''
            data_path = paths.get('exp1_driver_data')
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    LSG_dataset_path = paths.get('LSG_dataset_path')
    
    
    with strategy.scope():
        #load models
        #define conv layer
        conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                                 'conv_kernel_size':LN_params.get('conv_kernel_size'),
                                 'window_size':window_size}
        
        conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params,
                                                              load_weights_path = load_weights_path)
        
        #create dummy input for LN init
        dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
        dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                              dummy_input, num_steps, training = True)
        
        #define LN model
        LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                        load_weights_path = load_weights_path,
                                                        dummy_in = dummy_conv_out,
                                                        output_version = output_version)
        
        if output_version == 2:
            def compute_loss(loss_input, targets, class_weights, num_steps,
                             one_hot, global_batch_size):
                loss = classification_loss(loss_input, targets, class_weights,
                                               num_steps, one_hot,
                                               parallel = True)
                
                #scale loss
                scale_loss = tf.math.divide(tf.reduce_sum(loss), 
                                            tf.cast(global_batch_size,
                                                    dtype = tf.float32))
                return scale_loss
        else:
            from tools_and_functions import calculate_sample_weights
            def compute_loss(outputs, labels, class_weights, global_batch_size):
                out_sigmoid = outputs[0]
                loss_fcn = tf.keras.losses.BinaryCrossentropy(reduction = 
                                                              tf.keras.losses.Reduction.NONE)
                
                #calculate sample weights
                sample_weights = calculate_sample_weights(labels, class_weights,
                                                          one_hot)

                losses = loss_fcn(labels, out_sigmoid,
                                  sample_weight = tf.reshape(sample_weights,
                                                             shape = [len(sample_weights),1]))
                scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                            tf.cast(global_batch_size,dtype=tf.float32))
                
                return scale_loss
            
        #specify trainable variables
        conv_layer_1 = conv_layers_dict.get('layer_1')
        conv_layer_2 = conv_layers_dict.get('layer_2')
        train_var = conv_layer_1.trainable_variables +\
            conv_layer_2.trainable_variables +\
                LN_output_layer.trainable_variables
                
        #define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
    #train function
    def train_func(inputs, targets):
        with tf.GradientTape(persistent=True) as tape:
            [tape.watch(v) for v in train_var]
            
            #forward pass
            conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                         inputs, num_steps, training = True)
            LN_Hidden_out = LN_Hidden_layer(conv_out)
            outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
            #extract outputs
            if LN_output_layer.name == "LN_output_layer_v2":
                out_pred = outputs[2]
            else:
                out_pred = outputs[1]
            
            #compute loss
            if output_version == 2:
                loss_input = outputs
            else:
                loss_input = out_pred
            batch_size = tf.shape(targets)[0]
            loss = compute_loss(loss_input, targets, class_weights,
                                           num_steps, one_hot, batch_size)
                 
        #compute gradients for hidden layer using eprop
        Learning_signal = tape.gradient(loss, LN_Hidden_out[0])
        
        #compute eprop gradients            
        [hid_grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
            LN_params,
            Learning_signal,
            conv_out,LN_Hidden_out[0],
            LN_Hidden_out[1],LN_Hidden_out[2],
            zero_on_diagonal = False, decay_out = None)
       
        #grad for recurrent
        z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                     LN_Hidden_out[0][:,:-1]], axis=1)
        [hid_grad_rec, _, _, _] = LSNN_compute_loss_gradient(
            LN_params, Learning_signal, z_previous_step,
            LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
            decay_out = None)
        
        #compute standard gradients
        grads = tape.gradient(loss, train_var)
        
        #append grads
        grads.append(hid_grad_in)
        grads.append(hid_grad_rec)
        
        #apply gradients
        optimizer.apply_gradients([(g,v) for g,v in zip(grads,train_var +
                                                       LN_Hidden_layer.trainable_variables)])
        
        #save LSG input and learning signal tensors
        #create LSG input
        #get random integers totaling 10% of the batch size
        conv_out_shape = tf.shape(conv_out)
        LSG_input_spike = tf.reshape(conv_out,
                                     shape = (conv_out_shape[0],
                                              conv_out_shape[1],
                                              conv_out_shape[-3]*
                                              conv_out_shape[-2]*
                                              conv_out_shape[-1]))
        LSG_inputs = tf.concat([LSG_input_spike,
                                LN_Hidden_out[0]], axis = 2)
        
        return loss, targets, LSG_inputs, Learning_signal
        
    
    #distribution wrapper
    def dist_train_step(inputs, targets):
        per_replica_losses, per_replica_targets, per_replica_LSG_inputs, per_replica_Learning_signals = strategy.run(train_func,
                                args = (inputs, targets))
        
        if strategy.num_replicas_in_sync > 1:
            total_targets = tf.concat(per_replica_targets, axis = 0)
            total_LSG_inputs = tf.concat(per_replica_LSG_inputs.values, axis = 0)
            total_Learning_signals = tf.concat(per_replica_Learning_signals.values,
                                               axis = 0)
        else:
            total_targets = per_replica_targets
            total_LSG_inputs = per_replica_LSG_inputs
            total_Learning_signals = per_replica_Learning_signals
            
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis = None), total_targets, total_LSG_inputs, total_Learning_signals

    #read in experiment 1 data
    if transfer_learning == 1:
        from tools_and_functions import Load_all_driver_data
        dataset = Load_all_driver_data(data_path, one_hot = one_hot)
    else:
        dataset = Load_dataset(data_path, one_hot = one_hot)
    
    #calculate class weight from dataset
    class_weights = calculate_class_weights(dataset, one_hot)
    
    #create train test split
    #get labels
    y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                         dtype = object)
    x_indices = np.arange(len(y_targets))
    
    #get split indices
    if window_size == 1:
        train_split_frac = 0.8
    elif window_size == 2:
        train_split_frac = 0.75
    elif window_size == 3:
        train_split_frac = 0.66
    else:
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
    
    #initialize counter and dataset folder for LSG dataset creation
    LSG_data_count = 0
    if os.path.exists(LSG_dataset_path):
        shutil.rmtree(LSG_dataset_path)
    os.mkdir(LSG_dataset_path)
    os.mkdir(os.path.join(LSG_dataset_path, 'inputs'))
    os.mkdir(os.path.join(LSG_dataset_path, 'targets'))
    
    #main loop
    record_weights = 0
    training_losses = []
    LSG_inputs_list = []
    Learning_signals_list = []
    for epoch in range(max_epochs):
        epoch_loss = 0
        for step, (inputs, targets) in enumerate(train_dataset):
            
            #train step
            loss, targets, LSG_inputs, Learning_signals = dist_train_step(inputs, targets)
            LSG_inputs_list.append(LSG_inputs)
            Learning_signals_list.append(Learning_signals)
            
            #add to epoch loss
            epoch_loss += loss.numpy()
            
            #save LSG_inputs and learning signals
            if ((epoch+1) % 10) == 0:
                LSG_data_count += 1
                save_data(LSG_inputs_list,
                          os.path.join(LSG_dataset_path,
                                       'inputs', f"input_{LSG_data_count}.npz"))
                save_data(Learning_signals_list,
                          os.path.join(LSG_dataset_path,
                                       'targets', f"target_{LSG_data_count}.npz"))
                
                #reset lists
                LSG_inputs_list = []
                Learning_signals_list = []
            
        #save loss
        print(f'loss = {epoch_loss}')
        training_losses.append(epoch_loss)
            
        #test
        for step, (inputs, targets) in enumerate(test_dataset):            
            #forward pass
            conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                         inputs, num_steps, training = True)
            LN_Hidden_out = LN_Hidden_layer(conv_out)
            outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
            #extract outputs
            if LN_output_layer.name == "LN_output_layer_v2":
                out_pred = outputs[2]
            else:
                out_pred = outputs[1]
            
            #save output
            if step == 0:
                total_targets = targets
                total_outputs = out_pred.numpy()
            else:
                total_targets = tf.concat([total_targets,targets], axis = 0)
                total_outputs = tf.concat([total_outputs, out_pred.numpy()], 
                                          axis = 0)
                 
        #get metrics
        metrics = get_metrics(total_outputs, total_targets, one_hot = one_hot)
        
        #print achieved metrics
        print(f"Achieved Testing Accuracy: {metrics[0]}")
        print(f"Achieved Testing TPR:  {metrics[1]}")
        print(f"Achieved Testing TNR: {metrics[2]}")
        
        #check if satisfactory performance is met
        if metrics[0] >= 0.95 and metrics[1] >= 0.9:
            record_weights = 1
            
            #break from loop
            break
        
    #save LSG data if list is non-empty
    if len(LSG_inputs_list) > 0:
        LSG_data_count += 1
        save_data(LSG_inputs_list,
                  os.path.join(LSG_dataset_path,
                               'inputs', f"input_{LSG_data_count}.npz"))
        save_data(Learning_signals_list,
                  os.path.join(LSG_dataset_path,
                               'targets', f"target_{LSG_data_count}.npz"))
        
        #reset lists
        LSG_inputs_list = []
        Learning_signals_list = []
            
    #save weights if performance metrics were met
    if record_weights == 1:
        succeeded = 1
        print("performance criteria met, saving weights")
        
        save_weights(conv_layer_1.get_weights(), 
                     os.path.join(save_weights_path,
                                  'LN_conv_layer_1_weights.npz'))
        save_weights(conv_layer_2.get_weights(),
                     os.path.join(save_weights_path,
                                  'LN_conv_layer_2_weights.npz'))
        save_weights(LN_Hidden_layer.get_weights(),
                     os.path.join(save_weights_path,
                                  'LN_Hidden_weights.npz'))
        save_weights(LN_output_layer.get_weights(), 
                     os.path.join(save_weights_path,
                                  'LN_output_weights.npz'))
    else:
        succeeded = 0 
        print("performance criteria not met.")
        
    #making training loss plot
    num_updates = np.arange(1,len(training_losses)+1)
    plt.plot(num_updates,training_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Training Loss per Epoch')
    if transfer_learning == 1:
        plt.savefig(os.path.join(save_results_path, 'LNTrainingLossPerEpoch2.png'))
    else:
        plt.savefig(os.path.join(save_results_path, 'LNTrainingLossPerEpoch.png'))
    plt.close()
        
    return succeeded

def LSG_supervised_Learning(paths, params, transfer_learning):
    
    import tensorflow as tf
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from tools_and_functions import Load_LSG_Supervised_Learning_Dataset
    from tools_and_functions import get_split_dataset, save_weights
    from Models_and_Layers import create_LSG_model, get_output
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #unpack paths
    store_path = paths.get('LN_LSG_supervised_learning_store_path')
    save_weights_path = os.path.join(store_path, 'weights')
    save_results_path = os.path.join(store_path, 'results')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    if transfer_learning == 1:
        data_path = paths.get('LSG_dataset_2_path')
        load_weights_path = save_weights_path
    else:
        data_path = paths.get('LSG_dataset_path')
        load_weights_path = ""
        
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    SL_params = params.get('SL_params') 
    batch_size = SL_params.get('batch_size')
    learning_rate = SL_params.get('learning_rate')
    max_epochs = SL_params.get('epochs_train')
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    num_steps = LN_params.get('num_steps')
    MetaL_params.update({'num_steps':num_steps})
    
    #define dataset
    dataset = Load_LSG_Supervised_Learning_Dataset(data_path)
    
    with strategy.scope():
        #load models  
        #create dummy input
        single_data_point = next(iter(dataset))
        dummy_LSG_inputs = tf.zeros_like(tf.expand_dims(single_data_point[0],
                                                        axis = 0))
        
        #build model
        Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                               load_weights_path = load_weights_path,
                                                        dummy_in = dummy_LSG_inputs)
        
        #define LSG loss
        def compute_LSG_loss(Learning_signal, Learning_signal_pred, global_batch_size):
            LSG_loss_func = tf.keras.losses.MeanSquaredError(reduction = 
                                                             tf.keras.losses.Reduction.NONE)
            
            losses = LSG_loss_func(Learning_signal, Learning_signal_pred)
            
            scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                        tf.cast(global_batch_size, dtype = tf.float32))
            return scale_loss
        
        #specify trainable variables
        train_var = Hidden_LSG.trainable_variables + LSG_out.trainable_variables
        
        #define optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        
    #train function
    def train_func(inputs, targets):
        with tf.GradientTape() as tape:
            
            #forward pass TEST THIS
            Hidden_LSG_outs = Hidden_LSG(inputs)
            LS_pred = get_output(Hidden_LSG_outs[0], LSG_out,
                                 ol_training = True)
            
            #get loss
            this_batch_size = tf.shape(targets)[0]
            loss = compute_LSG_loss(targets, LS_pred, this_batch_size)
            
        #compute grads
        grads = tape.gradient(loss, train_var)
        
        #apply grads
        optimizer.apply_gradients([(g,v) for g,v in zip(grads, train_var)])
        
        return loss, targets
    
    #dist wrapper
    def dist_train_step(inputs, targets):
        
        per_replica_losses, per_replica_targets = strategy.run(train_func,
                                                               args = (inputs, targets))
        
        if strategy.num_replicas_in_sync > 1:
            total_targets = tf.concat(per_replica_targets.values, axis = 0)
        else:
            total_targets = per_replica_targets
            
        return(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis = None), total_targets)
    
    #create train test split
    #get targets
    y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                         dtype = object)
    x_indices = np.arange(len(y_targets))
    
    train_split_frac = 0.8
    x_tr_ind, x_te_ind, y_tr_ind, y_te_ind = train_test_split(x_indices,
                                                              y_targets,
                                                              test_size = (1-train_split_frac),
                                                              random_state = 42)
    
    train_dataset = get_split_dataset(dataset, x_tr_ind)
    train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    
    test_dataset = get_split_dataset(dataset, x_te_ind)
    test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    
    #main loop
    LSG_epoch_losses = []
    for epoch in range(max_epochs):
        LSG_epoch_loss = 0
        for step, (inputs, targets) in enumerate(train_dataset):
            
            #train step
            loss, targets = dist_train_step(inputs, targets)
            
            #add to loss counters
            LSG_epoch_loss += loss
            
        #save losses
        print(f'LSG loss = {LSG_epoch_loss}')
        LSG_epoch_losses.append(LSG_epoch_loss)
        
        #test
        for step, (inputs, targets) in enumerate(test_dataset):
            #get predictions
            Hidden_LSG_outs = Hidden_LSG(inputs)
            LS_pred = get_output(Hidden_LSG_outs[0], LSG_out,
                                 ol_training = True)
            
            #append to total targets and predictions
            if step == 0:
                total_targets = targets
                total_pred = LS_pred
            else:
                total_targets = tf.concat([total_targets, targets], axis = 0)
                total_pred = tf.concat([total_pred, LS_pred], axis = 0)
                
        #metrics
        MAE = tf.keras.metrics.MeanAbsoluteError(name = "mean_absolute_error",
                                                 dtype = None)
        MAE.update_state(total_pred, total_targets)
        test_MAE = MAE.result().numpy()
        print(f'Test MAE: {test_MAE}')
        
        if test_MAE <= 1e-3:
            break
        
    #record weights (Always record weights)
    print('Saving weights')
    
    save_weights(Hidden_LSG.get_weights(),
                     os.path.join(save_weights_path,
                                   'Hidden_LSG_weights.npz'))
    save_weights(LSG_out.get_weights(),
                 os.path.join(save_weights_path,
                              'LSG_out_weights.npz'))
        
    #plot LSG loss vs epoch
    epoch_enum = np.arange(1, len(LSG_epoch_losses)+1)
    plt.plot(epoch_enum, LSG_epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('LSG losses')
    plt.title('LSG Loss per Epoch')
    if transfer_learning == 1:
        plt.savefig(os.path.join(save_results_path, 'LSGTrainingLossPerEpoch2.png'))
    else:
        plt.savefig(os.path.join(save_results_path, 'LSGTrainingLossPerEpoch.png'))
    plt.close()
    
    finished = 1
    return finished
        

def LN_LSG_supervised_learning_with_eprop(paths, params):
    
    import tensorflow as tf
    import numpy as np
    import os
    import shutil
    from sklearn.model_selection import train_test_split
    from tools_and_functions import get_split_dataset
    from tools_and_functions import Load_dataset
    from tools_and_functions import calculate_class_weights
    from tools_and_functions import classification_loss, get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import save_weights
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import conv_forward_pass, get_output
    from Models_and_Layers import create_LSG_model
    import matplotlib.pyplot as plt
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    SL_params = params.get('SL_params')
    
    transfer_learning = SL_params.get('transfer_learning')
    if transfer_learning is None:
        transfer_learning = 0
    elif transfer_learning:
        custom_weights_path = SL_params.get('custom_weights_path')
    debug = SL_params.get('debug')
    if debug is None:
        debug = False
    batch_size = SL_params.get('batch_size')
    learning_rate = SL_params.get('learning_rate')
    max_epochs = SL_params.get('epochs_train')
    num_LSG_updates = SL_params.get('num_LSG_updates')
    num_times_train_LN = SL_params.get('num_times_train_LN')
    freeze_conv_layer_weights = SL_params.get('freeze_conv_layer_weights')
    if freeze_conv_layer_weights is None:
        freeze_conv_layer_weights = 0
    train_LN_LSNN_from_scratch = SL_params.get('train_LN_LSNN_from_scratch')
    if train_LN_LSNN_from_scratch is None:
        train_LN_LSNN_from_scratch = 0
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    window_size = MetaL_params.get('window_size')
    num_steps = LN_params.get('num_steps')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
    
    #unpack paths
    store_path = paths.get('LN_LSG_supervised_learning_store_path')
    save_weights_path = os.path.join(store_path, 'weights')
    save_results_path = os.path.join(store_path, 'results')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    if os.path.exists(save_weights_path) or custom_weights_path is not None:
        if transfer_learning != 1:
            shutil.rmtree(save_weights_path)
            os.mkdir(save_weights_path)
            load_LN_weights_path = ''
            data_path = paths.get('exp1_driver_data')
        else:
            if custom_weights_path is None:
                load_LN_weights_path = save_weights_path
            else:
                load_LN_weights_path = custom_weights_path
            data_path = paths.get('MetaL_driver_data')
    else:
        if transfer_learning == 1:
            # incorrect config
            print("Previous weights do not exist, can't do transfer learning")
            import sys
            sys.exit()
        else:
            os.mkdir(save_weights_path)
            if custom_weights_path is None:
                load_LN_weights_path = ''
            else:
                load_LN_weights_path = custom_weights_path
            data_path = paths.get('exp1_driver_data')
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
        
    for loop_ind in range(num_times_train_LN):
        print(f'Training loop pass: {loop_ind+1}')
        if loop_ind > 0:
            load_LSG_weights_path = save_weights_path
        else:
            load_LSG_weights_path = ''
    
        with strategy.scope():
            #load models
            #define conv layer
            conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                                     'conv_kernel_size':LN_params.get('conv_kernel_size'),
                                     'window_size':window_size}
            
            conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params,
                                                                  load_weights_path = load_LN_weights_path)
            
            #create dummy input for LN init
            dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
            dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                                  dummy_input, num_steps, training = True)
            
            #define LN model
            if train_LN_LSNN_from_scratch:
                LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                            dummy_in = dummy_conv_out,
                                                            output_version = output_version)
            else:
                LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                            load_weights_path = load_LN_weights_path,
                                                            dummy_in = dummy_conv_out,
                                                            output_version = output_version)
            
            #define LN loss
            if output_version == 2:
                def compute_LN_loss(loss_input, targets, class_weights, num_steps,
                                 one_hot, global_batch_size):
                    loss = classification_loss(loss_input, targets, class_weights,
                                                   num_steps, one_hot,
                                                   parallel = True)
                    
                    #scale loss
                    scale_loss = tf.math.divide(tf.reduce_sum(loss), 
                                                tf.cast(global_batch_size,
                                                        dtype = tf.float32))
                    return scale_loss
            else:
                from tools_and_functions import calculate_sample_weights
                def compute_LN_loss(outputs, labels, class_weights, global_batch_size):
                    out_sigmoid = outputs[0]
                    loss_fcn = tf.keras.losses.BinaryCrossentropy(reduction = 
                                                                  tf.keras.losses.Reduction.NONE)
                    
                    #calculate sample weights
                    sample_weights = calculate_sample_weights(labels, class_weights,
                                                              one_hot)
        
                    losses = loss_fcn(labels, out_sigmoid,
                                      sample_weight = tf.reshape(sample_weights,
                                                                 shape = [len(sample_weights),1]))
                    scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                                tf.cast(global_batch_size,dtype=tf.float32))
                    
                    return scale_loss
            
            #define LSG model
            #update meta l params with num steps
            MetaL_params.update({'num_steps':num_steps})
            
            #dummy input
            dummy_LSG_input_spike_shape = tf.shape(dummy_conv_out)
            dummy_LSG_input_spike = tf.reshape(dummy_conv_out,
                                               shape = (dummy_LSG_input_spike_shape[0],
                                                        dummy_LSG_input_spike_shape [1],
                                                        dummy_LSG_input_spike_shape[-3]*
                                                        dummy_LSG_input_spike_shape[-2]*
                                                        dummy_LSG_input_spike_shape[-1]))
            dummy_LSG_inputs = tf.concat([dummy_LSG_input_spike,
                                                     dummy_LN_Hidden_outs[0]], axis = 2)
            
            #build model
            Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                                            load_weights_path = load_LSG_weights_path,
                                                            dummy_in = dummy_LSG_inputs)
            
            #define LSG loss
            def compute_LSG_loss(Learning_signal, Learning_signal_pred, global_batch_size):
                LSG_loss_func = tf.keras.losses.MeanSquaredError(reduction = 
                                                                 tf.keras.losses.Reduction.NONE)
                
                losses = LSG_loss_func(Learning_signal, Learning_signal_pred)
                
                scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                            tf.cast(global_batch_size, dtype = tf.float32))
                return scale_loss
                
            #specify trainable variables
            conv_layer_1 = conv_layers_dict.get('layer_1')
            conv_layer_2 = conv_layers_dict.get('layer_2')
            if freeze_conv_layer_weights == 0:
                train_var = conv_layer_1.trainable_variables +\
                    conv_layer_2.trainable_variables +\
                        LN_output_layer.trainable_variables
            else:
                train_var = LN_output_layer.trainable_variables
                    
            #define optimizers
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            
        
        #train function
        def train_func(inputs, targets):
            with tf.GradientTape(persistent = True) as tape:
                # LN training
        
                #forward pass
                conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                             inputs, num_steps, training = True)
                LN_Hidden_out = LN_Hidden_layer(conv_out)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer,
                                     ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                
                #compute loss
                if output_version == 2:
                    loss_input = outputs
                else:
                    loss_input = out_pred
                batch_size = tf.shape(targets)[0]    
                LN_loss = compute_LN_loss(loss_input, targets, class_weights,
                                               num_steps, one_hot, batch_size)
            
            #compute learning signal using autodiff
            Learning_signal = tape.gradient(LN_loss, LN_Hidden_out[0])
                
            #compute eprop gradients            
            [hid_grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                LN_params,
                Learning_signal,
                conv_out,LN_Hidden_out[0],
                LN_Hidden_out[1],LN_Hidden_out[2],
                zero_on_diagonal = False, decay_out = None)
           
            #grad for recurrent
            z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                         LN_Hidden_out[0][:,:-1]], axis=1)
            [hid_grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                LN_params, Learning_signal, z_previous_step,
                LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
                decay_out = None)
            
            #compute standard gradients
            grads = tape.gradient(LN_loss, train_var)
            
            #append grads
            grads.append(hid_grad_in)
            grads.append(hid_grad_rec)
            
            #apply gradients
            optimizer_1.apply_gradients([(g,v) for g,v in zip(grads,train_var +
                                                           LN_Hidden_layer.trainable_variables)])
            
            for ii in range(num_LSG_updates):
                #train LSG five times for every 1 LN train
                with tf.GradientTape() as tape:
                    #get LSG prediction from LN spikes
                    conv_out_shape = tf.shape(conv_out)
                    LSG_input_spike = tf.reshape(conv_out,
                                                 shape = (conv_out_shape[0],
                                                          conv_out_shape[1],
                                                          conv_out_shape[-3]*
                                                          conv_out_shape[-2]*
                                                          conv_out_shape[-1]))
                    LSG_inputs = tf.concat([LSG_input_spike,
                                            LN_Hidden_out[0]], axis = 2)
                    Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
                    Learning_signal_pred = get_output(Hidden_LSG_outs[0], LSG_out,
                                                 ol_training = True)
                     
                    #compute LSG loss
                    batch_size = tf.shape(targets)[0]
                    LSG_loss = compute_LSG_loss(Learning_signal,
                                             Learning_signal_pred, batch_size)
                
                #compute LSG grads
                LSG_grads = tape.gradient(LSG_loss, Hidden_LSG.trainable_variables + 
                                                   LSG_out.trainable_variables)
            
                optimizer_2.apply_gradients([(g,v) for g,v in zip(LSG_grads,
                                            Hidden_LSG.trainable_variables + 
                                                   LSG_out.trainable_variables)])
            
            return LN_loss, LSG_loss, targets
        
        #distribution wrapper
        def dist_train_step(inputs, targets):
            per_replica_LN_losses, per_replica_LSG_losses, per_replica_targets = strategy.run(train_func, 
                                                                                              args = (inputs, targets))
            
            if strategy.num_replicas_in_sync > 1:
                total_targets = tf.concat(per_replica_targets, axis = 0)
            else:
                total_targets = per_replica_targets
            
            return(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_LN_losses,
                                   axis = None), strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                 per_replica_LSG_losses,
                                                                 axis = None), 
                                                                 total_targets)
        
        #read in experiment 1 data
        if transfer_learning == 1:
            from tools_and_functions import Load_all_driver_data
            dataset, dataset_length = Load_all_driver_data(data_path, one_hot = one_hot,
                                                           debug = debug)
        else:
            dataset, dataset_length = Load_dataset(data_path, one_hot = one_hot,
                                                   debug = debug)
        
        #calculate class weight from dataset
        class_weights = calculate_class_weights(dataset, one_hot)
        
        #create train test split
        #get labels
        y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                             dtype = object)
        x_indices = np.arange(len(y_targets))
        
        #get split indices
        if window_size == 1:
            train_split_frac = 0.8
        elif window_size == 2:
            train_split_frac = 0.75
        elif window_size == 3:
            train_split_frac = 0.66
        else:
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
                
        #main loop
        record_weights = 0
        if loop_ind == 0:
            LSG_epoch_losses = []
        LN_training_losses = []
        for epoch in range(max_epochs):
            print(f'Epoch: {epoch}')
            LN_epoch_loss = 0
            LSG_epoch_loss = 0
            for step, (inputs, targets) in enumerate(train_dataset):
                
                #train step
                LN_loss, LSG_loss, targets = dist_train_step(inputs, targets)
                
                #add to loss counters
                LN_epoch_loss += LN_loss
                LSG_epoch_loss += LSG_loss
                
            #save losses
            print(f'LN loss = {LN_epoch_loss}')
            print(f'LSG loss = {LSG_epoch_loss}')
            LN_training_losses.append(LN_epoch_loss)
            LSG_epoch_losses.append(LSG_epoch_loss)
                
            #test
            for step, (inputs, targets) in enumerate(test_dataset):
                    
                #forward pass
                conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                             inputs, num_steps, training = True)
                LN_Hidden_out = LN_Hidden_layer(conv_out)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                
                #save output
                if step == 0:
                    total_targets = targets
                    total_outputs = out_pred.numpy()
                else:
                    total_targets = tf.concat([total_targets,targets], axis = 0)
                    total_outputs = tf.concat([total_outputs, out_pred.numpy()], 
                                              axis = 0)
                     
            #get metrics
            metrics = get_metrics(total_outputs, total_targets, one_hot = one_hot)
            
            #print achieved metrics
            print(f"Achieved Testing Accuracy: {metrics[0]}")
            print(f"Achieved Testing TPR:  {metrics[1]}")
            print(f"Achieved Testing TNR: {metrics[2]}")
        
        # check metrics
        record_weights = 0    
        if loop_ind == 0:
            best_metrics = metrics
            record_weights = 1
        else:
        #if the overall accuracy and TNR rating are better (the best classifier is one that never mislabels doing nothing as doing something)
            if metrics[0] > best_metrics[0] and metrics[2] > best_metrics[2]:
                best_metrics = metrics
                record_weights = 1
                
        #save weights if performance metrics were met
        if record_weights == 1:
            succeeded = 1
            print("performance criteria met, saving weights")
            
            save_weights(conv_layer_1.get_weights(), 
                         os.path.join(save_weights_path,
                                      'LN_conv_layer_1_weights.npz'))
            save_weights(conv_layer_2.get_weights(),
                         os.path.join(save_weights_path,
                                      'LN_conv_layer_2_weights.npz'))
            save_weights(LN_Hidden_layer.get_weights(),
                         os.path.join(save_weights_path,
                                      'LN_Hidden_weights.npz'))
            save_weights(LN_output_layer.get_weights(), 
                         os.path.join(save_weights_path,
                                      'LN_output_weights.npz'))
            save_weights(Hidden_LSG.get_weights(),
                         os.path.join(save_weights_path,
                                       'Hidden_LSG_weights.npz'))
            save_weights(LSG_out.get_weights(),
                         os.path.join(save_weights_path,
                                      'LSG_out_weights.npz'))
                                      
            #making training loss plot
            if not debug:
                num_updates = np.arange(1,len(LN_training_losses)+1)
                plt.plot(num_updates,LN_training_losses)
                plt.xlabel('Epoch')
                plt.ylabel('Loss Value')
                plt.title('Training Loss per Epoch')
                if transfer_learning == 1:
                    plt.savefig(os.path.join(save_results_path, f'LNLossPerEpoch2_{loop_ind}.png'))
                else:
                    plt.savefig(os.path.join(save_results_path, f'LNLossPerEpoch_{loop_ind}.png'))
                plt.close()
        else:
            succeeded = 0 
            print("performance criteria not met.")
        
    #plot LSG loss vs epoch
    if not debug:
        epoch_enum = np.arange(1, len(LSG_epoch_losses)+1)
        plt.plot(epoch_enum, LSG_epoch_losses)
        plt.xlabel('Epochs')
        plt.ylabel('LSG losses')
        plt.title('LSG Loss per Epoch')
        if transfer_learning == 1:
            plt.savefig(os.path.join(save_results_path, f'LSGLossPerEpoch2_{loop_ind}.png'))
        else:
            plt.savefig(os.path.join(save_results_path, f'LSGLossPerEpoch_{loop_ind}.png'))
        plt.close()
        
    return succeeded

def train_LN_with_LSG(paths, params):
    #import
    import tensorflow as tf
    import numpy as np
    import os
    from tools_and_functions import get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import save_weights
    from Models_and_Layers import create_conv_layer, create_LN_model
    from Models_and_Layers import conv_forward_pass, get_output
    from Models_and_Layers import create_LSG_model
    import matplotlib.pyplot as plt
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    SL_params = params.get('SL_params')
    
    #define paths
    tfs = SL_params.get('tfs')
    transfer_learning = SL_params.get('transfer_learning')
    if transfer_learning is None:
        transfer_learning = 0
    elif transfer_learning:
        custom_weights_path = SL_params.get('custom_weights_path')
    store_path = paths.get('LN_LSG_supervised_learning_store_path')
    save_weights_path = os.path.join(store_path, 'weights_LSG_trained')
    save_results_path = os.path.join(store_path, 'results_LSG_trained')
    if transfer_learning==1:
        if custom_weights_path is not None:
            load_LN_weights_path = custom_weights_path
            load_LSG_weights_path = custom_weights_path
        else:
            load_LN_weights_path = os.path.join(store_path, 'weights')
            load_LSG_weights_path = os.path.join(store_path, 'weights')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
    if transfer_learning == 1:
        data_path = paths.get('MetaL_driver_data')
    else:
        data_path = paths.get('exp1_driver_data')
        
    #unpack remaining params
    debug = SL_params.get('debug')
    if debug is None:
        debug = False
    train_or_test = SL_params.get('train_or_test')
    batch_size = SL_params.get('batch_size')
    learning_rate = SL_params.get('learning_rate')
    max_epochs = SL_params.get('epochs_train')
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    window_size = MetaL_params.get('window_size')
    num_steps = LN_params.get('num_steps')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
    if tfs == 1:
        LN_params.update({'use_random_LSNN_weights':True})
    else:
        LN_params.update({'use_random_LSNN_weights':False})
        
    with strategy.scope():
        #load models
        #define conv layer
        conv_layer_params = {'conv_filters':LN_params.get('conv_filters'),
                                 'conv_kernel_size':LN_params.get('conv_kernel_size'),
                                 'window_size':window_size}
        
        conv_layers_dict, LIF_layers_dict = create_conv_layer(conv_layer_params,
                                                              load_weights_path = load_LN_weights_path)
        
        #create dummy input for LN init
        dummy_input = tf.convert_to_tensor(np.zeros([1,19,996*window_size,1]))
        dummy_conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                              dummy_input, num_steps, training = True)
        
        #define LN model
        LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                        load_weights_path = load_LN_weights_path,
                                                        dummy_in = dummy_conv_out,
                                                        output_version = output_version)
        
        #define LSG model
        #update meta l params with num steps
        MetaL_params.update({'num_steps':num_steps})
        
        #dummy input
        dummy_LSG_input_spike_shape = tf.shape(dummy_conv_out)
        dummy_LSG_input_spike = tf.reshape(dummy_conv_out,
                                           shape = (dummy_LSG_input_spike_shape[0],
                                                    dummy_LSG_input_spike_shape [1],
                                                    dummy_LSG_input_spike_shape[-3]*
                                                    dummy_LSG_input_spike_shape[-2]*
                                                    dummy_LSG_input_spike_shape[-1]))
        dummy_LSG_inputs = tf.concat([dummy_LSG_input_spike,
                                                 dummy_LN_Hidden_outs[0]], axis = 2)
        
        #build model
        Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                                        load_weights_path = load_LSG_weights_path,
                                                        dummy_in = dummy_LSG_inputs)
        
        #define optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        if train_or_test == "train":
            from tools_and_functions import classification_loss
            #define loss
            if output_version == 2:
                def compute_loss(outputs, labels, class_weights, num_steps,
                                 one_hot, global_batch_size):
                    loss = classification_loss(outputs, labels, class_weights,
                                                   num_steps, one_hot,
                                                   parallel = True)
                    
                    #scale loss
                    scale_loss = tf.math.divide(tf.reduce_sum(loss), 
                                                tf.cast(global_batch_size,
                                                        dtype = tf.float32))
                    return scale_loss
                                      
            else:
                from tools_and_functions import calculate_sample_weights
                def compute_loss(outputs, labels, class_weights, global_batch_size):
                    out_sigmoid = outputs[0]
                    loss_fcn = tf.keras.losses.BinaryCrossentropy(reduction = 
                                                                  tf.keras.losses.Reduction.NONE)
                    
                    #calculate sample weights
                    sample_weights = calculate_sample_weights(labels, class_weights,
                                                              one_hot)
        
                    losses = loss_fcn(labels, out_sigmoid,
                                      sample_weight = tf.reshape(sample_weights,
                                                                 shape = [len(sample_weights),1]))
                    scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                                tf.cast(global_batch_size,dtype=tf.float32))
                    
                    return scale_loss
    
    if train_or_test == "train":
        #train function
        from tools_and_functions import SGD
        from Models_and_Layers import LSNN_func
        
        #define surrogate weights
        LN_Hidden_weights = LN_Hidden_layer.trainable_variables
        LN_W_Surro = [tf.identity(LN_Hidden_weights[0]),
                   tf.identity(LN_Hidden_weights[1])]
        
        def train_func(inputs, targets):
            with tf.GradientTape() as tape:
                tape.watch(LN_W_Surro)
            
                #forward pass
                conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                             inputs, num_steps, training = True)
                LN_Hidden_out = LSNN_func(conv_out, LN_W_Surro, LN_params)
                
                #get LSG prediction from LN spikes
                conv_out_shape = tf.shape(conv_out)
                LSG_input_spike = tf.reshape(conv_out,
                                             shape = (conv_out_shape[0],
                                                      conv_out_shape[1],
                                                      conv_out_shape[-3]*
                                                      conv_out_shape[-2]*
                                                      conv_out_shape[-1]))
                LSG_inputs = tf.concat([LSG_input_spike,
                                        LN_Hidden_out[0]], axis = 2)
                Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
                Learning_signal = get_output(Hidden_LSG_outs[0], LSG_out,
                                             ol_training = True)
                
                #compute eprop gradients            
                [hid_grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                    LN_params,
                    Learning_signal,
                    conv_out,LN_Hidden_out[0],
                    LN_Hidden_out[1],LN_Hidden_out[2],
                    zero_on_diagonal = False, decay_out = None)
               
                #grad for recurrent
                z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                             LN_Hidden_out[0][:,:-1]], axis=1)
                [hid_grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                    LN_params, Learning_signal, z_previous_step,
                    LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
                    decay_out = None)
                
                #apply gradients using SGD
                grad_list = [hid_grad_in, hid_grad_rec]
                LN_W_Surro_new = SGD(LN_W_Surro, grad_list, learning_rate)
                
                #second pass
                LN_Hidden_out = LSNN_func(conv_out, LN_W_Surro_new, LN_params)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer,
                                 ol_training = True)
                
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                    
                #get metrics
                metrics = get_metrics(out_pred, targets, one_hot = one_hot)
#                print(f'Accuracy: {metrics[0]}')
#                print(f'TPR: {metrics[1]}')
#                print(f'TNR: {metrics[2]}')
                    
                #compute LN loss
                batch_size = tf.shape(targets)[0]
                loss = compute_loss(outputs, targets, class_weights, num_steps,
                                one_hot, batch_size)
                
            #calc LSG grads
            grads = tape.gradient(loss, Hidden_LSG.trainable_variables +\
                                         LSG_out.trainable_variables)
                
            if grads is None:
                print("Gradient not working")
                
            #apply grads
            optimizer.apply_gradients([(g,v) for g,v in zip(grads, Hidden_LSG.trainable_variables +\
                                         LSG_out.trainable_variables)])
                
            return loss, metrics
                
        
    else:
        #train function
        def train_func(inputs, targets):
            
            for ii in range(10):
                #forward pass
                conv_out = conv_forward_pass(conv_layers_dict, LIF_layers_dict,
                                             inputs, num_steps, training = True)
                LN_Hidden_out = LN_Hidden_layer(conv_out)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer,
                                     ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                    
                #get metrics
                metrics = get_metrics(out_pred, targets, one_hot = one_hot)
                print(f'Accuracy: {metrics[0]}')
                print(f'TPR: {metrics[1]}')
                print(f'TNR: {metrics[2]}')
                    
                #get LSG prediction from LN spikes
                conv_out_shape = tf.shape(conv_out)
                LSG_input_spike = tf.reshape(conv_out,
                                             shape = (conv_out_shape[0],
                                                      conv_out_shape[1],
                                                      conv_out_shape[-3]*
                                                      conv_out_shape[-2]*
                                                      conv_out_shape[-1]))
                LSG_inputs = tf.concat([LSG_input_spike,
                                        LN_Hidden_out[0]], axis = 2)
                Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
                Learning_signal = get_output(Hidden_LSG_outs[0], LSG_out,
                                             ol_training = True)
                
                #compute eprop gradients            
                [hid_grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                    LN_params,
                    Learning_signal,
                    conv_out,LN_Hidden_out[0],
                    LN_Hidden_out[1],LN_Hidden_out[2],
                    zero_on_diagonal = False, decay_out = None)
               
                #grad for recurrent
                z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                             LN_Hidden_out[0][:,:-1]], axis=1)
                [hid_grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                    LN_params, Learning_signal, z_previous_step,
                    LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
                    decay_out = None)
                
                #create grad list
                grads = []
                grads.append(hid_grad_in)
                grads.append(hid_grad_rec)
                
                #apply grads
                optimizer.apply_gradients([(g,v) for g,v in zip(grads,
                                                               LN_Hidden_layer.trainable_variables)])
            
            return metrics
    
    #distribution wrapper
    if train_or_test == "train":
        def dist_train_step(inputs, targets):
            per_replica_loss, per_replica_metrics = strategy.run(train_func, args = (inputs, targets))
            
            if strategy.num_replicas_in_sync > 1:
                total_metrics = tf.math.reduce_mean(per_replica_metrics.values)
            else:
                total_metrics = per_replica_metrics
                
            return (strategy.reduce(tf.distribute.ReduceOp.SUM,
                                   per_replica_loss, axis = None), total_metrics)
    else:
        def dist_train_step(inputs, targets):
            per_replica_metrics = strategy.run(train_func, args = (inputs, targets))
            
            if strategy.num_replicas_in_sync > 1:
                total_metrics = strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                                per_replica_metrics, axis = None)
            else:
                total_metrics = per_replica_metrics
                
            return total_metrics
    
    #read in exp2 data
    if transfer_learning == 1:
        from tools_and_functions import Load_all_driver_data
        dataset, dataset_length = Load_all_driver_data(data_path, one_hot = one_hot,
                                                       debug = debug)
    else:
        from tools_and_functions import Load_dataset
        dataset, dataset_length = Load_dataset(data_path, one_hot = one_hot,
                                               debug = debug)
        
    if train_or_test == "train":
        from tools_and_functions import calculate_class_weights
        class_weights = calculate_class_weights(dataset, one_hot)
    
    dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size)
    
    #main loop
    if train_or_test == "train":
        loss_hist = []
    for epoch in range(max_epochs):
        print(f'Epoch: {epoch}')
        
        for step, (inputs, targets) in enumerate(dataset):
            
            if train_or_test == "train":
                #train step
                loss, metrics = dist_train_step(inputs, targets)
                loss_hist.append(loss)
                
            else:
                #train step
                metrics = dist_train_step(inputs, targets)
            
            if step == 0 and epoch == 0:
                metrics_hist = np.expand_dims(metrics, axis = 0)
            else:
                metrics_hist = np.concatenate([metrics_hist, 
                                               np.expand_dims(metrics, axis = 0)],
                                              axis = 0)
            
    #save weights
    conv_layer_1 = conv_layers_dict.get('layer_1')
    conv_layer_2 = conv_layers_dict.get('layer_2')
    save_weights(conv_layer_1.get_weights(), 
                         os.path.join(save_weights_path,
                                      'LN_conv_layer_1_weights.npz'))
    save_weights(conv_layer_2.get_weights(),
                 os.path.join(save_weights_path,
                              'LN_conv_layer_2_weights.npz'))
    save_weights(LN_Hidden_layer.get_weights(),
                 os.path.join(save_weights_path,
                              'LN_Hidden_weights.npz'))
    save_weights(LN_output_layer.get_weights(), 
                 os.path.join(save_weights_path,
                              'LN_output_weights.npz'))
    save_weights(Hidden_LSG.get_weights(),
                 os.path.join(save_weights_path,
                               'Hidden_LSG_weights.npz'))
    save_weights(LSG_out.get_weights(),
                 os.path.join(save_weights_path,
                              'LSG_out_weights.npz'))
    
    #plot metrics vs step
    steps = np.arange(1, len(metrics_hist)+1)
    fig, axs = plt.subplots(np.shape(metrics_hist[0])[0], figsize = (10,6))
    fig.suptitle('Training Metrics Per Batch of Data')
    #create plots
    axs[0].plot(steps, metrics_hist[:,0])
    axs[0].set_title('Accuracy')
    axs[0].set(xlabel = 'Batches', ylabel = 'Percent')
    
    axs[1].plot(steps, metrics_hist[:,1])
    axs[1].set_title('TPR')
    axs[1].set(xlabel = 'Batches', ylabel = 'Percent')
    
    axs[2].plot(steps, metrics_hist[:,2])
    axs[2].set_title('TNR')
    axs[2].set(xlabel = 'Batches', ylabel = 'Percent')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_results_path, 'MetricsPerBatch.png'))
    plt.close()
    
    if train_or_test == "train":
        #also plot loss
        epochs = np.arange(1, len(loss_hist)+1)
        plt.plot(epochs, loss_hist)
        plt.xlabel("Batches")
        plt.ylabel("LN Loss")
        plt.title("LN Loss Per Batch")
        plt.savefig(os.path.join(save_results_path, 'LN_Loss_per_Batch.png'))
        plt.close()
    
    succeeded = 1
    return succeeded


def LN_LSG_SupLearn_eprop_LSNN_only(paths, params, transfer_learning):
    
    import tensorflow as tf
    import numpy as np
    import os
    import shutil
    from sklearn.model_selection import train_test_split
    from tools_and_functions import get_split_dataset
    from tools_and_functions import Load_dataset
    from tools_and_functions import calculate_class_weights
    from tools_and_functions import classification_loss, get_metrics
    from tools_and_functions import LSNN_compute_loss_gradient
    from tools_and_functions import save_weights
    from Models_and_Layers import create_LN_model
    from Models_and_Layers import get_output
    from Models_and_Layers import create_LSG_model
    import matplotlib.pyplot as plt
    
    #define distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    #print devices tf can access
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    #unpack paths
    store_path = paths.get('LN_LSG_SupLearn_LSNN_only_store_path')
    save_weights_path = os.path.join(store_path, 'weights')
    save_results_path = os.path.join(store_path, 'results')
    if not os.path.exists(store_path):
        os.mkdir(store_path)
    if os.path.exists(save_weights_path):
        if transfer_learning != 1:
            shutil.rmtree(save_weights_path)
            os.mkdir(save_weights_path)
            load_LN_weights_path = ''
            data_path = paths.get('exp1_driver_data')
        else:
            load_LN_weights_path = save_weights_path
            data_path = paths.get('MetaL_drivers_data')
    else:
        if transfer_learning == 1:
            # incorrect config
            print("Previous weights do not exist, can't do transfer learning")
            import sys
            sys.exit()
        else:
            os.mkdir(save_weights_path)
            load_LN_weights_path = ''
            data_path = paths.get('exp1_driver_data')
    if not os.path.exists(save_results_path):
        os.mkdir(save_results_path)
        
    #define params
    LN_params = params.get('LN_params')
    MetaL_params = params.get('MetaL_params')
    SL_params = params.get('SL_params')
    
    batch_size = SL_params.get('batch_size')
    learning_rate = SL_params.get('learning_rate')
    max_epochs = SL_params.get('epochs_train')
    num_LSG_updates = SL_params.get('num_LSG_updates')
    num_times_train_LN = SL_params.get('num_times_train_LN')
    shuffle_buffer_size = MetaL_params.get('shuffle_buffer_size')
    window_size = MetaL_params.get('window_size')
    num_steps = LN_params.get('num_steps')
    output_version = LN_params.get('output_version')
    if output_version == 2:
        one_hot = True
    else:
        one_hot = False
        
    for loop_ind in range(num_times_train_LN):
        print(f'Training loop pass: {loop_ind+1}')
        if loop_ind > 0:
            load_LSG_weights_path = save_weights_path
        else:
            load_LSG_weights_path = ''
            
        with strategy.scope():
            
            # create dummy_input
            dummy_input = tf.convert_to_tensor(np.zeros([1,996,19]),
                                               dtype = tf.float32)
            
            #load models (LSNN only)
            #define LN model
            LN_Hidden_layer, LN_output_layer, dummy_LN_Hidden_outs = create_LN_model(LN_params,
                                                            load_weights_path = load_LN_weights_path,
                                                            dummy_in = dummy_input,
                                                            output_version = output_version)
            
            #define LN loss
            if output_version == 2:
                def compute_LN_loss(loss_input, targets, class_weights, num_steps,
                                 one_hot, global_batch_size):
                    loss = classification_loss(loss_input, targets, class_weights,
                                                   num_steps, one_hot,
                                                   parallel = True)
                    
                    #scale loss
                    scale_loss = tf.math.divide(tf.reduce_sum(loss), 
                                                tf.cast(global_batch_size,
                                                        dtype = tf.float32))
                    return scale_loss
            else:
                from tools_and_functions import calculate_sample_weights
                def compute_LN_loss(outputs, labels, class_weights, global_batch_size):
                    out_sigmoid = outputs[0]
                    loss_fcn = tf.keras.losses.BinaryCrossentropy(reduction = 
                                                                  tf.keras.losses.Reduction.NONE)
                    
                    #calculate sample weights
                    sample_weights = calculate_sample_weights(labels, class_weights,
                                                              one_hot)
        
                    losses = loss_fcn(labels, out_sigmoid,
                                      sample_weight = tf.reshape(sample_weights,
                                                                 shape = [len(sample_weights),1]))
                    scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                                tf.cast(global_batch_size,dtype=tf.float32))
                    
                    return scale_loss
                
            #define LSG model            
            #dummy input
            dummy_LSG_inputs = tf.concat([dummy_input,
                                         dummy_LN_Hidden_outs[0]],
                                         axis = 2)
            
            #build model
            Hidden_LSG, LSG_out = create_LSG_model(MetaL_params,
                                                            load_weights_path = load_LSG_weights_path,
                                                            dummy_in = dummy_LSG_inputs)
            
            #define LSG loss
            def compute_LSG_loss(Learning_signal, Learning_signal_pred, global_batch_size):
                LSG_loss_func = tf.keras.losses.MeanSquaredError(reduction = 
                                                                 tf.keras.losses.Reduction.NONE)
                
                losses = LSG_loss_func(Learning_signal, Learning_signal_pred)
                
                scale_loss = tf.math.divide(tf.reduce_sum(losses),
                                            tf.cast(global_batch_size, dtype = tf.float32))
                return scale_loss
                
            #specify trainable variables
            train_var = LN_output_layer.trainable_variables
                    
            #define optimizers
            optimizer_1 = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            optimizer_2 = tf.keras.optimizers.Adam(learning_rate = learning_rate)
            
        #train function
        def train_func(inputs, targets):
            with tf.GradientTape(persistent = True) as tape:
                # LN training
        
                #forward pass
                inputs = tf.cast(inputs, dtype = tf.float32)
                input_shape = tf.shape(inputs)
                input_reshape = tf.reshape(inputs, shape = (input_shape[0],
                                                            input_shape[2],
                                                            input_shape[1]))
                LN_Hidden_out = LN_Hidden_layer(input_reshape)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer,
                                     ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                
                #compute loss
                if output_version == 2:
                    loss_input = outputs
                else:
                    loss_input = out_pred
                batch_size = tf.shape(targets)[0]    
                LN_loss = compute_LN_loss(loss_input, targets, class_weights,
                                               996, one_hot, batch_size)
            
            #compute learning signal using autodiff
            Learning_signal = tape.gradient(LN_loss, LN_Hidden_out[0])
                
            #compute eprop gradients            
            [hid_grad_in, e_trace, eps_v, eps_a] = LSNN_compute_loss_gradient(
                LN_params,
                Learning_signal,
                input_reshape,LN_Hidden_out[0],
                LN_Hidden_out[1],LN_Hidden_out[2],
                zero_on_diagonal = False, decay_out = None)
           
            #grad for recurrent
            z_previous_step = tf.concat([tf.zeros_like(LN_Hidden_out[0][:,0])[:, None],
                                         LN_Hidden_out[0][:,:-1]], axis=1)
            [hid_grad_rec, _, _, _] = LSNN_compute_loss_gradient(
                LN_params, Learning_signal, z_previous_step,
                LN_Hidden_out[0], LN_Hidden_out[1], LN_Hidden_out[2], zero_on_diagonal = True,
                decay_out = None)
            
            #compute standard gradients
            grads = tape.gradient(LN_loss, train_var)
            
            #append grads
            grads.append(hid_grad_in)
            grads.append(hid_grad_rec)
            
            #apply gradients
            optimizer_1.apply_gradients([(g,v) for g,v in zip(grads,train_var +
                                                           LN_Hidden_layer.trainable_variables)])
            
            for ii in range(num_LSG_updates):
                #train LSG five times for every 1 LN train
                with tf.GradientTape() as tape:
                    #get LSG prediction from LN spikes
                    LSG_inputs = tf.concat([input_reshape,
                                            LN_Hidden_out[0]], axis = 2)
                    Hidden_LSG_outs = Hidden_LSG(LSG_inputs)
                    Learning_signal_pred = get_output(Hidden_LSG_outs[0], LSG_out,
                                                 ol_training = True)
                     
                    #compute LSG loss
                    batch_size = tf.shape(targets)[0]
                    LSG_loss = compute_LSG_loss(Learning_signal,
                                             Learning_signal_pred, batch_size)
                
                #compute LSG grads
                LSG_grads = tape.gradient(LSG_loss, Hidden_LSG.trainable_variables + 
                                                   LSG_out.trainable_variables)
            
                optimizer_2.apply_gradients([(g,v) for g,v in zip(LSG_grads,
                                            Hidden_LSG.trainable_variables + 
                                                   LSG_out.trainable_variables)])
            
            return LN_loss, LSG_loss, targets
        
        #distribution wrapper
        def dist_train_step(inputs, targets):
            per_replica_LN_losses, per_replica_LSG_losses, per_replica_targets = strategy.run(train_func, 
                                                                                              args = (inputs, targets))
            
            if strategy.num_replicas_in_sync > 1:
                total_targets = tf.concat(per_replica_targets, axis = 0)
            else:
                total_targets = per_replica_targets
            
            return(strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_LN_losses,
                                   axis = None), strategy.reduce(tf.distribute.ReduceOp.SUM,
                                                                 per_replica_LSG_losses,
                                                                 axis = None), 
                                                                 total_targets)
                                                                 
        #read in data
        if transfer_learning == 1:
            from tools_and_functions import Load_all_driver_data
            dataset, dataset_length = Load_all_driver_data(data_path, one_hot = one_hot)
        else:
            dataset, dataset_length = Load_dataset(data_path, one_hot = one_hot)
            
        #calculate class weight from dataset
        class_weights = calculate_class_weights(dataset, one_hot)
        
        #create train test split
        #get labels
        y_targets = np.array([target.numpy() for _, target in iter(dataset)],
                             dtype = object)
        x_indices = np.arange(len(y_targets))
        
        #get split indices
        if window_size == 1:
            train_split_frac = 0.8
        elif window_size == 2:
            train_split_frac = 0.75
        elif window_size == 3:
            train_split_frac = 0.66
        else:
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
        
        #main loop
        record_weights = 0
        if loop_ind == 0:
            LSG_epoch_losses = []
        LN_training_losses = []
        for epoch in range(max_epochs):
            print(f'Epoch: {epoch}')
            LN_epoch_loss = 0
            LSG_epoch_loss = 0
            for step, (inputs, targets) in enumerate(train_dataset):
                
                #train step
                LN_loss, LSG_loss, targets = dist_train_step(inputs, targets)
                
                #add to loss counters
                LN_epoch_loss += LN_loss
                LSG_epoch_loss += LSG_loss
                
            #save losses
            print(f'LN loss = {LN_epoch_loss}')
            print(f'LSG loss = {LSG_epoch_loss}')
            LN_training_losses.append(LN_epoch_loss)
            LSG_epoch_losses.append(LSG_epoch_loss)
                
            #test
            for step, (inputs, targets) in enumerate(test_dataset):
                    
                #forward pass
                input_shape = tf.shape(inputs)
                input_reshape = tf.reshape(inputs, shape = (input_shape[0],
                                                            input_shape[2],
                                                            input_shape[1]))
                LN_Hidden_out = LN_Hidden_layer(input_reshape)
                outputs = get_output(LN_Hidden_out[0], LN_output_layer, ol_training = True)
                #extract outputs
                if LN_output_layer.name == "LN_output_layer_v2":
                    out_pred = outputs[2]
                else:
                    out_pred = outputs[1]
                
                #save output
                if step == 0:
                    total_targets = targets
                    total_outputs = out_pred.numpy()
                else:
                    total_targets = tf.concat([total_targets,targets], axis = 0)
                    total_outputs = tf.concat([total_outputs, out_pred.numpy()], 
                                              axis = 0)
                     
            #get metrics
            metrics = get_metrics(total_outputs, total_targets, one_hot = one_hot)
            
            #print achieved metrics
            print(f"Achieved Testing Accuracy: {metrics[0]}")
            print(f"Achieved Testing TPR:  {metrics[1]}")
            print(f"Achieved Testing TNR: {metrics[2]}")
            
            # check metrics
            record_weights = 0    
            if loop_ind == 0:
                best_metrics = metrics
                record_weights = 1
            else:
            #if the overall accuracy and TNR rating are better (the best classifier is one that never mislabels doing nothing as doing something)
                if metrics[0] > best_metrics[0] and metrics[2] > best_metrics[2]:
                    best_metrics = metrics
                    record_weights = 1
                    
            #save weights if performance metrics were met
            if record_weights == 1:
                succeeded = 1
                print("performance criteria met, saving weights")
                
                save_weights(LN_Hidden_layer.get_weights(),
                             os.path.join(save_weights_path,
                                          'LN_Hidden_weights.npz'))
                save_weights(LN_output_layer.get_weights(), 
                             os.path.join(save_weights_path,
                                          'LN_output_weights.npz'))
                save_weights(Hidden_LSG.get_weights(),
                             os.path.join(save_weights_path,
                                           'Hidden_LSG_weights.npz'))
                save_weights(LSG_out.get_weights(),
                             os.path.join(save_weights_path,
                                          'LSG_out_weights.npz'))
                                          
                #making training loss plot
                num_updates = np.arange(1,len(LN_training_losses)+1)
                plt.plot(num_updates,LN_training_losses)
                plt.xlabel('Epoch')
                plt.ylabel('Loss Value')
                plt.title('Training Loss per Epoch')
                if transfer_learning == 1:
                    plt.savefig(os.path.join(save_results_path, f'LNLossPerEpoch2_{loop_ind}.png'))
                else:
                    plt.savefig(os.path.join(save_results_path, f'LNLossPerEpoch_{loop_ind}.png'))
                plt.close()
            else:
                succeeded = 0 
                print("performance criteria not met.")
        
    #plot LSG loss vs epoch
    epoch_enum = np.arange(1, len(LSG_epoch_losses)+1)
    plt.plot(epoch_enum, LSG_epoch_losses)
    plt.xlabel('Epochs')
    plt.ylabel('LSG losses')
    plt.title('LSG Loss per Epoch')
    if transfer_learning == 1:
        plt.savefig(os.path.join(save_results_path, f'LSGLossPerEpoch2_{loop_ind}.png'))
    else:
        plt.savefig(os.path.join(save_results_path, f'LSGLossPerEpoch_{loop_ind}.png'))
    plt.close()
        
    return succeeded


def make_meta_learning_training_plots(paths):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    #read in paths
    #define storage path
    store_path = paths.get('store_path')
    saveFigFilePath = paths.get('results_path')
    
    #read in metric data
    metrics_pre_data = np.load(os.path.join(store_path, 'training_metrics_pre.npy'))
    metrics_post_data = np.load(os.path.join(store_path, 'training_metrics_post.npy'))
    
    #create separate plots for each individual driver and store them in the fig
    #save file path
    size_metrics_data = np.shape(metrics_post_data)
    num_epochs = np.arange(1,size_metrics_data[2]+1)
    for num_driver in range(size_metrics_data[1]):
        fig, axs = plt.subplots(size_metrics_data[0], figsize = (10,6))
        fig.suptitle(f'Average Training Metrics for Driver {num_driver + 1} per Epoch')
        #create plots using matplotlib 
        axs[0].plot(num_epochs, metrics_pre_data[0,num_driver,:],label = 'Pre E-prop',
                    linestyle = 'dashed')
        axs[0].plot(num_epochs, metrics_post_data[0,num_driver,:],label = 'Post E-prop')
        axs[0].set_title('Accuracy')
        axs[0].set(xlabel = 'Epochs', ylabel = 'Percent')
        axs[0].legend()
        axs[1].plot(num_epochs, metrics_pre_data[1,num_driver,:],label = 'Pre E-prop',
                    linestyle = 'dashed')
        axs[1].plot(num_epochs, metrics_post_data[1,num_driver,:],label = 'Post E-prop')
        axs[1].set_title('TPR')
        axs[1].set(xlabel = 'Epochs', ylabel = 'Percent')
        axs[1].legend()
        axs[2].plot(num_epochs, metrics_pre_data[2,num_driver,:],label = 'Pre E-prop TNR',
                    linestyle = 'dashed')
        axs[2].plot(num_epochs, metrics_post_data[2,num_driver,:],label = 'Post E-prop TNR')
        axs[2].set_title('TNR')
        axs[2].set(xlabel = 'Epochs', ylabel = 'Percent')
        axs[2].legend()

        fig.tight_layout()
        plt.savefig(os.path.join(saveFigFilePath, f'Driver{num_driver+1}MetricsPerEpoch.png'))
        plt.close()
        
    #read in outer loop loss data
    loss_data = np.load(os.path.join(store_path, 'training_loss.npy'))
    
    #outer loop loss plot
    loss_data_shape = np.shape(loss_data)
    num_epochs = np.arange(1,loss_data_shape[0]+1)
    plt.plot(num_epochs,loss_data)
    plt.xlabel('Outer Loop Epochs')
    plt.ylabel('Loss Value')
    plt.title('Outer Loop Loss per Epoch')
    plt.savefig(os.path.join(saveFigFilePath, 'OuterLoopLossPerEpoch.png'))
    plt.close()
    
def initialize(paths, params, train_config, reinitialize):
    from tools_and_functions import save_dict
    import shutil
    import os
        
    #define paths    
    base_store_path = paths.get('base_store_path')
    store_path = paths.get('store_path')
    results_path = os.path.join(store_path,'results')
    paths.update({'results_path':results_path})
    weights_path = os.path.join(store_path,'weights')
    paths.update({'weights_path':weights_path})
        
    if reinitialize == 1:
        #sort out params
        LN_params = params.get('LN_params')
        MetaL_params = params.get('MetaL_params')
        
        if os.path.exists(base_store_path):
            if os.path.exists(store_path):
                #delete storage
                shutil.rmtree(store_path)
          
        else:
            #create base directory
            os.mkdir(base_store_path)
            
        #recreate directories
        os.mkdir(store_path)
        os.mkdir(results_path)
        os.mkdir(weights_path)
        
        
        if train_config == 2:
            #copy final exp1 trained LN weights to LN model weights
            source = paths.get('LN_Pretrain_Final_weights_path')
            dest = weights_path
            for file_name in os.listdir(source):
                src = os.path.join(source, file_name)
                dst = os.path.join(dest, file_name)
                if os.path.isfile(src):
                    shutil.copy(src,dst)
                
        #reinitialize counters
        saveMetricNum = 0
        saveOutLoopLossNum = 0
        counters = {'saveMetricNum':saveMetricNum, 'saveOutLoopLossNum':saveOutLoopLossNum}
        
        #save dictionaries
        save_dict(os.path.join(store_path,'LN_params.txt'), LN_params)
        save_dict(os.path.join(store_path, 'MetaL_parameters.txt'), MetaL_params)
        save_dict(os.path.join(store_path, 'counters.txt'), counters)
        save_dict(os.path.join(store_path, 'paths.txt'), paths)
    
    return paths
    
    
    
        
        
        
        
    
    