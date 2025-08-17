# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:38:23 2023

@author: nalmrb
"""

def main(main_params={}):

    #import
    import os
    import tensorflow as tf
    import numpy as np
    import define_paths
    
    tf.random.set_seed(7)
    
    print("num GPUs available: ", len(tf.config.list_physical_devices('GPU')))
    #tf.debugging.set_log_device_placement(True)
    
    #set flags
    initial_training             = main_params.get("initial_training", 1)
    quantization                 = main_params.get("quantization", 1)
    few_shot_learning            = main_params.get("few_shot_learning", 1)
    num_loops                    = main_params.get('num_loops',10)
    compare_inf_times_and_energy = main_params.get("compare_inf_times_and_energy", 1)
    use_exp1_data                = main_params.get("use_exp1_data", 0)
    debug                        = main_params.get("debug",1)
    only_certain_iter            = main_params.get("only_certain_iter", [])
    run_cpu                      = main_params.get("run_cpu", 1)
    energyusage_debug            = main_params.get("energyusage_debug", 0)
    learn_comp_vals              = main_params.get("learn_comp_vals", [0]) #[0, 0.25, 0.5, 0.75, 0.9]
    chanInds                     = main_params.get("chanInds" , list(range(19)))
    max_epochs_tuple             = main_params.get("max_epochs", (125, 150, 25))
    if debug:
        print("DEBUG FLAG SET")
        num_loops = 1
        learn_comp_vals = [0]
    if run_cpu:
        print("RUNNING CPU, SET RUN_CPU = 0 TO RUN ON AKIDA")
    
    #define paths
    paths = define_paths.main(use_exp1_data)
    bootstrap_results_path = paths.get('bootstrap_results_path')
    #replace results_path if a results_path is included in the input params dict
    if "results_path" in main_params.keys():
        bootstrap_results_path = main_params.get("results_path")
    #create bootstrap results path if doesn't exist
    if not os.path.exists(bootstrap_results_path):
        os.mkdir(bootstrap_results_path)
    
    #define common params
    shuffle_buffer_size = 500
    batch_size          = 1024 #512
    one_hot             = False
    network             = 'CNN'
    metric_target       = 0.90
    num_total_drivers   = 11
    num_train_drivers   = 8
    
    res_file_path = os.path.join(bootstrap_results_path, 'results.txt')
    res_file = open(res_file_path, 'w')
    res_file.write('Bootstrap Results\n\n')
    #add param values
    res_file.write(f'initial_training: {initial_training}\n')
    res_file.write(f'quantization: {quantization}\n')
    res_file.write(f'few_shot_learning; {few_shot_learning}\n')
    res_file.write(f'num_loops: {num_loops}\n')
    res_file.write(f'compare_inf_times_and_energy: {compare_inf_times_and_energy}\n')
    res_file.write(f'use_exp1_data: {use_exp1_data}\n')
    res_file.write(f'debug: {debug}\n')
    res_file.write(f'only_certain_iter: {only_certain_iter}\n')
    res_file.write(f'run_cpu: {run_cpu}\n')
    res_file.write(f'energyusage_debug: {energyusage_debug}\n')
    res_file.write(f'learn_comp_vals: {learn_comp_vals}\n')
    res_file.write(f'chanInds: {chanInds}\n')
    res_file.write(f'max_epochs_tuple: {max_epochs_tuple}\n\n')
    res_file.close()
    
    final_metrics_hist                         = np.zeros([3*2, num_loops])
    edge_learn_ind_driver_prelim_metrics_hist  = np.zeros([3,num_total_drivers-num_train_drivers,num_loops])
    edge_learn_ind_driver_final_metrics_hist   = np.zeros([3,num_total_drivers-num_train_drivers,num_loops])
    ind_driver_convergence_epochs_hist         = np.zeros([num_total_drivers-num_train_drivers, num_loops])
    ind_driver_inf_times_and_energy_hist       = np.zeros([6,num_total_drivers-num_train_drivers, num_loops])
    
    params = {'shuffle_buffer_size':shuffle_buffer_size,
              'batch_size':batch_size,
              'one_hot':one_hot,
              'network':network,
              'res_file_path':res_file_path,
              'few_shot_learning':few_shot_learning,
              'use_exp1_data':use_exp1_data,
              'num_train_drivers':num_train_drivers,
              'metric_target':metric_target,
              'debug':debug,
              'run_cpu':run_cpu,
              'chanInds':chanInds}
    
    for loop_iter in range(num_loops):
        res_file = open(res_file_path, 'a')
        res_file.write(f'Iteration: {loop_iter+1}\n')
        res_file.write('\n')
        res_file.close()
        if len(only_certain_iter) > 0:
            if (loop_iter+1) not in only_certain_iter:
                print(f'skipping loop {loop_iter + 1}')
                continue
        loop_iter_str = str(loop_iter+1)
        results_path = os.path.join(bootstrap_results_path, f'{loop_iter_str}')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        models_path = os.path.join(results_path, 'models')
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        paths.update({'results_path':results_path,
                      'models_path':models_path})
        params.update({'bootstrap_idx':loop_iter})
                                        
        #train keras model on exp 2 data (weight refinement)
        if initial_training == 1:
            #update params
            stage = 'initial_training'
            data_filepath = paths.get('bootstrap_data_path')
            save_fig_path = os.path.join(results_path, 'Exp2_Refining')
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)
    
            if debug:
                max_epochs = 3 #60
            else:
                max_epochs = max_epochs_tuple[0]
            
            params.update({'stage':stage, 'data_filepath':data_filepath,
                           'max_epochs':max_epochs})   
            
            #import
            from tools import Sup_Learning_EEG_Data_Keras
            from tools import plot_metrics_from_hist
            
            net_2, train_hist_2, eval_score_2, skip = Sup_Learning_EEG_Data_Keras(paths, params)
            
            #record final score
            res_file = open(res_file_path, 'a')
            res_file.write(f'Initial trained model final scores: acc = {eval_score_2[1]}, TPR = {eval_score_2[2]}, TNR = {eval_score_2[3]}\n\n')
            res_file.close()
            
            plot_metrics_from_hist(train_hist_2, save_fig_path)
            
            #save results
            final_metrics_hist[0,loop_iter] = eval_score_2[1] #train_hist_2.history['accuracy'][-1]
            final_metrics_hist[1,loop_iter] = eval_score_2[2] #train_hist_2.history['TPR'][-1]
            final_metrics_hist[2,loop_iter] = eval_score_2[3] #train_hist_2.history['TNR'][-1]
            
            
        if quantization == 1:
            
            ### TO DO: add code to load net based on selected options
            
            #update params
            stage = 'quantized_retraining'
            data_filepath = paths.get('bootstrap_data_path')
            save_fig_path = os.path.join(results_path, 'Quantize_Recoup')
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)
                
            if debug:
                max_epochs = 3 #75
            else:
                max_epochs = max_epochs_tuple[1]
            
            params.update({'max_epochs':max_epochs,
                           'stage':stage, 'data_filepath':data_filepath})
            
            #quantize model
            from tools import quantize_model
            quant_ints = [8, 8, 1]
            net_quantized, quantized_score = quantize_model(net_2, params, quant_ints)
            res_file = open(res_file_path, 'a')
            # res_file.write(f'quantized score: acc = {quantized_score[1]}, TPR = {quantized_score[2]}, TNR = {quantized_score[3]}\n')
            # res_file.write('\n')
            
            # thresh = .05 #percent
            # if (((train_hist_2.history['accuracy'][-1] - quantized_score[1]) >= thresh) or 
            # (train_hist_2.history['TPR'][-1] - quantized_score[2]) >= thresh or
            # (train_hist_2.history['TNR'][-1] - quantized_score[3]) >= thresh):
                
            #do quantized aware training
            from tools import Sup_Learning_EEG_Data_Keras
            from tools import plot_metrics_from_hist
            
            net_quantized_tuned, train_hist_3, eval_score_3, skip = Sup_Learning_EEG_Data_Keras(paths, params, net_quantized)
            res_file.write(f'Quantized model final scores: acc = {eval_score_3[1]}, TPR = {eval_score_3[2]}, TNR = {eval_score_3[3]}\n\n')
            
            plot_metrics_from_hist(train_hist_3, save_fig_path)
            
            #save results
            final_metrics_hist[3,loop_iter] = eval_score_3[1] #train_hist_3.history['accuracy'][-1]
            final_metrics_hist[4,loop_iter] = eval_score_3[2] #train_hist_3.history['TPR'][-1]
            final_metrics_hist[5,loop_iter] = eval_score_3[3] #train_hist_3.history['TNR'][-1]
            
            res_file.close()
        
        #quantize keras model and do on-chip edge learning
        if few_shot_learning == 1:
            
            #import stuff
            import numpy as np
            from cnn2snn import convert
            from tools import get_dataset
            from tools import plot_metrics, estimate_spikes, estimate_num_neurons
            from tools import edge_learning, get_prelim_performance
            from akida import AkidaUnsupervised
            from akida import FullyConnected
            from tools import get_inf_times_and_energy
            #from math import ceil
            
            #update params
            stage = 'few_shot_learning'
            data_filepath = paths.get('bootstrap_data_path')
    
            if debug:
                max_epochs = 2
            else:
                max_epochs = max_epochs_tuple[2]
            estimate_neurons = 0
            
            params.update({'stage':stage, 'data_filepath':data_filepath,
                           'max_epochs':max_epochs})
            
            #convert model
            net_akida = convert(net_quantized_tuned)
                
            # try running parameter grid
            weights_to_try_list = [1.2] #[0.15, 0.7, 1.2, 1.8, 2.5]
            num_neurons_to_try_list = [1000] #500 #[100, 500, 1500, 2500, 5000]
            
            for weight_to_try in weights_to_try_list:
                for num_neurons_to_try in num_neurons_to_try_list:
                    
                    #load dataset
                    group_model_flag = 0
                    params.update({'group_model_flag':group_model_flag})
                    datasets, dataset_lengths, class_weights = get_dataset(params)
                    num_learning_iter = len(datasets)
                    max_count = len(learn_comp_vals)       
                            
                    for learn_iter in range(num_learning_iter):
                        for count in range(max_count):
                            dataset = datasets[learn_iter]
                            dataset_length = dataset_lengths[learn_iter]
                            res_file = open(res_file_path, 'a')
                            res_file.write(f'Driver {learn_iter+1}\n\n')
                            res_file.close()
                            print(f'Driver {learn_iter+1}')
                    
                            print('adding new layer..')
                            #get rid of last layer
                            net_akida.pop_layer()
                    
                            #estimate weights
                            spike_est = estimate_spikes(net_akida, dataset)
                            
                            #try out different num_neurons_per_class values
                            if estimate_neurons:
                                neurons_per_class = [
                                2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
                                100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000]
                                
                                weights_mults = [0.1, 0.5, 1, 1.2, 1.5, 2, 2.5]
                                
                                #estimate neurons
                                est_neurons_params = {'dataset_length':dataset_length, 'neurons_per_class':neurons_per_class,
                                          'weights_mults':weights_mults,
                                          'save_path':results_path,
                                          'spike_est':spike_est}
                                estimate_num_neurons(net_akida, dataset, est_neurons_params)
                                
                            #edge learning
                            #define parameters
                            weight_mult = weight_to_try
                            num_weights = int(weight_mult * spike_est)
                            num_neurons_per_class = num_neurons_to_try
                        
                            #replace last layer with layer with binary weights in real model
                            num_classes = 2
                        
                            layer_fc = FullyConnected(name='akida_edge_layer',
                                                      units=num_classes * num_neurons_per_class,
                                                      activation=False)
                            
                            #add akida layer
                            net_akida.add(layer_fc)
                        
                            #compile net to use akida edge learning
                            learn_comp_val = learn_comp_vals[count]
                            print(f'trying learn_comp = {learn_comp_val}...')
                            net_akida.compile(optimizer=AkidaUnsupervised(num_weights = num_weights,
                                                                          num_classes = num_classes,
                                                                          learning_competition = learn_comp_val))
                            res_file = open(res_file_path, 'a')
                            res_file.write(f'trying learn_comp: {learn_comp_val}\n\n')
                            res_file.close()
                            
                            #set up save fig dir 
                            few_shot_results_path = os.path.join(results_path, 'few_shot_learning')
                            if not os.path.exists(few_shot_results_path):
                                os.mkdir(few_shot_results_path)
                            if group_model_flag:
                                save_fig_path = os.path.join(few_shot_results_path, 'group_model')
                            else:
                                save_fig_path = os.path.join(few_shot_results_path, f'driver_{learn_iter + 1}')
                            if not os.path.exists(save_fig_path):
                                os.mkdir(save_fig_path)
                        
                            #get preliminary performance
                            prelim_metrics = get_prelim_performance(net_akida, dataset, params)
                            #save prelim metrics
                            prelim_met_file = open(os.path.join(save_fig_path, 'prelim_metrics.txt'), 'w')
                            prelim_met_file.write(f'Acc: {prelim_metrics[0]}\n')
                            prelim_met_file.write(f'TPR: {prelim_metrics[1]}\n')
                            prelim_met_file.write(f'TNR: {prelim_metrics[2]}\n')
                            prelim_met_file.close()
                            
                            #save prelim performance
                            edge_learn_ind_driver_prelim_metrics_hist[0,learn_iter,loop_iter] = prelim_metrics[0]
                            edge_learn_ind_driver_prelim_metrics_hist[1,learn_iter,loop_iter] = prelim_metrics[1]
                            edge_learn_ind_driver_prelim_metrics_hist[2,learn_iter,loop_iter] = prelim_metrics[2]
                            
                            #edge learn   
                            net_akida, metrics = edge_learning(net_akida, dataset, params)
                            #save edge learning metrics
                            metrics_npy = np.array(metrics)
                            np.save(os.path.join(save_fig_path, 'metrics.npy'), metrics_npy)
                    
                            #plot
                            plot_metrics(metrics, save_fig_path)
                                
                            #save results
                            edge_learn_ind_driver_final_metrics_hist[0,learn_iter,loop_iter] = metrics[0][-1]
                            edge_learn_ind_driver_final_metrics_hist[1,learn_iter,loop_iter] = metrics[1][-1]
                            edge_learn_ind_driver_final_metrics_hist[2,learn_iter,loop_iter] = metrics[2][-1]
                            
                            #get epochs until convergence
                            acc_ge_0p95_bool = metrics_npy[0] >= metric_target
                            tpr_ge_0p95_bool = metrics_npy[1] >= metric_target
                            tnr_ge_0p95_bool = metrics_npy[2] >= metric_target
                            
                            bool_stack = np.stack([acc_ge_0p95_bool, tpr_ge_0p95_bool,
                                                   tnr_ge_0p95_bool])
                            bool_sum = bool_stack.sum(axis = 0)
                            
                            if len(np.where(bool_sum == 3)[0]) == 0:
                                print('Edge Learning did not converge')
                                epoch_converged = -1
                            else:
                                epoch_converged = np.where(bool_sum == 3)[0][0]
                            
                            ind_driver_convergence_epochs_hist[learn_iter, loop_iter] = epoch_converged + 1
                                
                            # #save model
                            # net_akida.save(os.path.join(save_fig_path, 'net_akida.fbz'))
                            # #save dict rep too
                            # import pickle
                            # with open(os.path.join(save_fig_path,'net_akida_dict.pkl'),'wb') as f:
                            #     pickle.dump(net_akida.to_dict(),f)
                            
                            #if only ind drivers
                            if epoch_converged > -1:
                                
                                #get inference time and energy
                                if compare_inf_times_and_energy:
                                    
                                    #unbatch then rebatch to get entire
                                    #dataset in one batch
                                    dataset = dataset.unbatch()
                                    dataset = dataset.batch(dataset_length)
                                    
                                    #compare akida network and trained
                                    #quantized netowrk
                                    nets = [net_akida, net_quantized_tuned]
                                    
                                    #send to get inf times and energy function
                                    results = get_inf_times_and_energy(nets,
                                                                       dataset,
                                                                       paths,
                                                                       save_path=save_fig_path)
                                    
                                    #save results to be averaged later
                                    ind_driver_inf_times_and_energy_hist[0,learn_iter,loop_iter] = results.get('akida_time_epoch')
                                    ind_driver_inf_times_and_energy_hist[1,learn_iter,loop_iter] = results.get('akida_energy_epoch')
                                    ind_driver_inf_times_and_energy_hist[2,learn_iter,loop_iter] = results.get('CPU_time_epoch')
                                    ind_driver_inf_times_and_energy_hist[3,learn_iter,loop_iter] = results.get('CPU_energy_epoch_adjusted')
                                    ind_driver_inf_times_and_energy_hist[4,learn_iter,loop_iter] = results.get('CPU_energy_epoch_total')
                                    ind_driver_inf_times_and_energy_hist[5,learn_iter,loop_iter] = results.get('CPU_energy_epoch_baseline')
                                
                                #break from learn iter loop
                                break
                            
                            
    #save data
    np.save(os.path.join(bootstrap_results_path, 'Ind_Model_prelim_metrics.npy'), edge_learn_ind_driver_prelim_metrics_hist)
    np.save(os.path.join(bootstrap_results_path, 'Ind_Model_final_metrics.npy'), edge_learn_ind_driver_final_metrics_hist)
    np.save(os.path.join(bootstrap_results_path, 'Ind_Model_Convergence_Epochs.npy'), ind_driver_convergence_epochs_hist)
    np.save(os.path.join(bootstrap_results_path, 'Ind_Model_inf_times_and_energies.npy'), ind_driver_inf_times_and_energy_hist)
    np.save(os.path.join(bootstrap_results_path, 'Group_Model_final_metrics.npy'), final_metrics_hist)
    
            
    #calculate and record final metrics
    grand_ave = np.mean(final_metrics_hist, axis = 1)
    std_dev = np.std(final_metrics_hist, axis = 1)
    
    ind_driver_prelim_metrics_shape  = np.shape(edge_learn_ind_driver_prelim_metrics_hist)
    grand_ave_ind_driver_prelim_met  = np.mean(np.reshape(edge_learn_ind_driver_prelim_metrics_hist,
                                                          [ind_driver_prelim_metrics_shape[0],
                                                           ind_driver_prelim_metrics_shape[1]*ind_driver_prelim_metrics_shape[2]]),
                                               axis = -1)
    std_dev_ind_driver_prelim_met    = np.std(np.reshape(edge_learn_ind_driver_prelim_metrics_hist,
                                                          [ind_driver_prelim_metrics_shape[0],
                                                           ind_driver_prelim_metrics_shape[1]*ind_driver_prelim_metrics_shape[2]]),
                                               axis = -1)
    ind_driver_final_metrics_shape   = np.shape(edge_learn_ind_driver_final_metrics_hist)
    grand_ave_ind_driver_final_met   = np.mean(np.reshape(edge_learn_ind_driver_final_metrics_hist,
                                                          [ind_driver_final_metrics_shape[0],
                                                           ind_driver_final_metrics_shape[1]*ind_driver_final_metrics_shape[2]]),
                                               axis = -1)
    std_dev_ind_driver_final_met     = np.std(np.reshape(edge_learn_ind_driver_final_metrics_hist,
                                                          [ind_driver_final_metrics_shape[0],
                                                           ind_driver_final_metrics_shape[1]*ind_driver_final_metrics_shape[2]]),
                                              axis = -1)
    
    #ind model
    #calculate and record epochs to convergence
    #get shape of convergence epochs array
    ind_driver_convergence_epochs_shape = np.shape(ind_driver_convergence_epochs_hist)
    num_zero_ind_model = 0
    for driver_idx in range(ind_driver_convergence_epochs_shape[0]):
        
        #get indices of any zeros
        zero_idx = np.where(ind_driver_convergence_epochs_hist[driver_idx,:] == 0)[0]
        if type(zero_idx) is list or type(zero_idx) is tuple:
                zero_idx = zero_idx[0]
        num_zero = len(zero_idx)
        num_zero_ind_model += num_zero
        
        #remove zero values
        if num_zero > 0:
            if driver_idx == 0:
                ind_driver_convergence_epochs_total = np.delete(ind_driver_convergence_epochs_hist[driver_idx,:],
                                                      zero_idx)
            else:
                ind_driver_convergence_epochs_total = np.concatenate([ind_driver_convergence_epochs_total,
                                                                      np.delete(ind_driver_convergence_epochs_hist[driver_idx,:],
                                                                                                            zero_idx)])
        else:
            if driver_idx == 0:
                ind_driver_convergence_epochs_total = ind_driver_convergence_epochs_hist[driver_idx,:]
            else:
                ind_driver_convergence_epochs_total = np.concatenate([ind_driver_convergence_epochs_total,
                                                                      ind_driver_convergence_epochs_hist[driver_idx,:]])
                
    grand_ave_ind_driver_convergence_epochs = np.mean(ind_driver_convergence_epochs_total,
                                                      axis = -1)
    std_dev_ind_driver_convergence_epochs   = np.std(ind_driver_convergence_epochs_total,
                                                     axis = -1)
    
    if compare_inf_times_and_energy:
        #calculate mean of inference times and energy consumption between
        #akida net on akida chip and quantized network on cpu
        orig_shape = np.shape(ind_driver_inf_times_and_energy_hist)
        ind_driver_inf_t_and_e_grnd_ave = np.mean(np.reshape(ind_driver_inf_times_and_energy_hist,
                                                             [orig_shape[0], orig_shape[1]*orig_shape[2]]),
                                                  axis = -1)
        ind_driver_inf_t_and_e_std = np.std(np.reshape(ind_driver_inf_times_and_energy_hist,
                                                             [orig_shape[0], orig_shape[1]*orig_shape[2]]),
                                                  axis = -1)
    
    #record
    res_file = open(res_file_path, 'a')
    res_file.write('\n\n\n')
    res_file.write('Grand Average and Standard Deviation of final metrics\n')
    res_file.write('Grand Average (std dev)\n')
    res_file.write('Initial Training:\n')
    res_file.write(f'Accuracy: {grand_ave[0]} ({std_dev[0]}), ')
    res_file.write(f'TPR: {grand_ave[1]} ({std_dev[1]}), ')
    res_file.write(f'TNR: {grand_ave[2]} ({std_dev[2]})\n')
    res_file.write('Quantization Aware Retraining:\n')
    res_file.write(f'Accuracy: {grand_ave[3]} ({std_dev[3]}), ')
    res_file.write(f'TPR: {grand_ave[4]} ({std_dev[4]}), ')
    res_file.write(f'TNR: {grand_ave[5]} ({std_dev[5]})\n')
    res_file.write('Few Shot Learning, Ind Driver: Prelim:\n')
    res_file.write(f'Accuracy: {grand_ave_ind_driver_prelim_met[0]} ({std_dev_ind_driver_prelim_met[0]}), ')
    res_file.write(f'TPR: {grand_ave_ind_driver_prelim_met[1]} ({std_dev_ind_driver_prelim_met[1]}), ')
    res_file.write(f'TNR: {grand_ave_ind_driver_prelim_met[2]} ({std_dev_ind_driver_prelim_met[2]})\n')
    res_file.write('Few Shot Learning, Ind Driver: Final:\n')
    res_file.write(f'Accuracy: {grand_ave_ind_driver_final_met[0]} ({std_dev_ind_driver_final_met[0]}), ')
    res_file.write(f'TPR: {grand_ave_ind_driver_final_met[1]} ({std_dev_ind_driver_final_met[1]}), ')
    res_file.write(f'TNR: {grand_ave_ind_driver_final_met[2]} ({std_dev_ind_driver_final_met[2]})\n')
    res_file.write(f'Number of iterations not converged: Ind Model: {num_zero_ind_model}\n')
    res_file.write(f'Few Shot Learning, Ind Drivers 90% Convergence Epochs: {grand_ave_ind_driver_convergence_epochs} ({std_dev_ind_driver_convergence_epochs})\n')
    if compare_inf_times_and_energy:
        #record inf times and energy mean and std for all ind driver data
        res_file.write(f'Ind Driver single epoch inference times (s): akida chip - {ind_driver_inf_t_and_e_grnd_ave[0]} ({ind_driver_inf_t_and_e_std[0]}), CPU - {ind_driver_inf_t_and_e_grnd_ave[2]} ({ind_driver_inf_t_and_e_std[2]})\n')
        res_file.write(f'Ind Driver single epoch inference energy (mJ): akida chip - {ind_driver_inf_t_and_e_grnd_ave[1]} ({ind_driver_inf_t_and_e_std[1]})\n')
        res_file.write(f'Ind Driver single epoch inference energy (mJ): CPU (baseline) - {ind_driver_inf_t_and_e_grnd_ave[5]} ({ind_driver_inf_t_and_e_std[5]}), CPU (total) - {ind_driver_inf_t_and_e_grnd_ave[4]} ({ind_driver_inf_t_and_e_std[4]}), CPU (adjusted) - {ind_driver_inf_t_and_e_grnd_ave[3]} ({ind_driver_inf_t_and_e_std[3]})\n')
    res_file.close()
    
        
    if energyusage_debug:
        
        from pyJoules.energy_meter import measure_energy
        from pyJoules.handler.csv_handler import CSVHandler
        from tools import get_dataset
        import akida
        import time
        from cnn2snn import convert
        from os.path import join, exists
        from os import remove
        import csv
        
        #set up params
        data_filepath = paths.get('bootstrap_data_path')
    
        params.update({'data_filepath':data_filepath})
        dataset, dataset_length, class_weights = get_dataset(params)
        
        results_path = paths.get('results_path')
        
        csv_path = join(results_path, 'pyJoules_energy_measurement_results.csv')
        #remove old csv if exists 
        if exists(csv_path):
            remove(csv_path)
        csv_handler = CSVHandler(csv_path)
                    
        #define function for keras model
        @measure_energy(handler = csv_handler)
        def get_net_predictions(net, inputs):
                        
                pred = net(inputs)
    
                return pred
            
        @measure_energy(handler = csv_handler)
        def get_baseline_energy():
            
            time.sleep(1)
            done = 1
            
            return done
            
        #set up akida for energy measurement
        #convert model
        net_akida = convert(net_quantized_tuned)
        #map net to device
        devices = akida.devices()
        device = devices[0]
        net_akida.map(device)
        #enable power measurement
        device.soc.power_measurement_enabled = True
            
        for step, (inputs, targets) in enumerate(dataset):
            
            #test pyjoules
            done = get_baseline_energy()
            pred = get_net_predictions(net_2, inputs)
            
            #test akida
            time_start = time.time()
            #pred = net_akida.predict_classes(inputs.numpy(), num_classes = 2)
            _ = net_akida.forward(inputs.numpy())
            time_end = time.time()
            
            inf_time = time_end - time_start
            
            break
        
        csv_handler.save_data()
        
        #get akida energy
        akida_energy_statistics = net_akida.statistics
        akida_framerate = akida_energy_statistics.fps
        akida_energy_per_frame = akida_energy_statistics.energy #mJ/frame
        akida_power_mW = akida_energy_per_frame * akida_framerate #mJ/frame * frame/sec = mJ/sec = mW
        akida_energy_mJ = akida_power_mW * inf_time
        print(f'inf time akida: {inf_time} seconds')
        print(f'inf power akida: {akida_power_mW} mW')
        print(f'inf total energy akida: {akida_energy_mJ} mJ')
        
        #read in csv
        energy_used = []
        durations = []
        count = -1
        with open(join(results_path, 'pyJoules_energy_measurement_results.csv'),
                       newline = '') as csvfile:
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
        #get inf power
        inf_power = energy_used[-1] / 1000 / durations[-1]
        #subtract baseline power
        inf_power_actual = inf_power - baseline_power
        #get actual energy consumption in milli Joules
        inf_energy_actual = inf_power_actual * durations[-1]
        
        print(f'inf time cpu: {durations[-1]} seconds')
        print(f'inf power cpu: {inf_power_actual} mW')
        print(f'inf total energy cpu: {inf_energy_actual} mJ')
        
#run main function of script if running from top-level
if __name__=='__main__':
    import sys
    sys.exit(main())