# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 20:39:59 2023

@author: nalmrb
"""

def main(main_params={}):

    #import
    import Define_paths_exp3
    import os
    import tensorflow as tf
    import numpy as np
    
    tf.random.set_seed(7)
    print("num GPUs available: ", len(tf.config.list_physical_devices('GPU')))
    
    #flags
    train                        = main_params.get("initial_training", 1)
    quantize                     = main_params.get("quantization", 1)
    few_shot_learning            = main_params.get("few_shot_learning", 1)
    compare_inf_times_and_energy = main_params.get("compare_inf_times_and_energy", 1)
    bootstrap_num                = main_params.get("num_loops", 10)
    debug                        = main_params.get("debug", 0)
    only_certain_iter            = main_params.get("only_certain_iter", [])
    run_cpu                      = main_params.get("run_cpu", 1)
    learn_comp_vals              = main_params.get("learn_comp_vals", [0]) #[0, 0.25, 0.5, 0.75, 0.9]
    chanInds                     = main_params.get("chanInds", list(range(19)))
    max_epochs_tuple             = main_params.get("max_epochs",(125,150,25))
    if debug:
        print("DEBUG FLAG SET")
        bootstrap_num = 1
        learn_comp_vals = [0]
    if run_cpu:
        print("RUNNING CPU, SET RUN_CPU = 0 TO RUN ON AKIDA")
    
    #define paths
    paths = Define_paths_exp3.main()
    #replace results_path if a results_path is included in the input params dict
    if "results_path" in main_params.keys():
        paths.update({"results_path_init":main_params.get("results_path")})
    #create results path if it doesn't exist
    if not os.path.exists(paths.get("results_path_init")):
        os.mkdir(paths.get("results_path_init"))
    
    #define common params
    shuffle_buffer_size = 500
    batch_size          = 1024 #512 value from first run
    one_hot             = False
    network             = 'CNN'
    num_train_drivers   = 8
    metric_target       = 0.9
        
    params = {'shuffle_buffer_size':shuffle_buffer_size,
              'batch_size':batch_size,
              'one_hot':one_hot,
              'network':network,
              'debug':debug,
              'run_cpu':run_cpu,
              'num_train_drivers':num_train_drivers,
              'metric_target':metric_target,
              'chanInds':chanInds}
        
    #start results txt file
    res_file_path = os.path.join(paths.get('results_path_init'), 'results.txt')
    res_file = open(res_file_path, 'w')
    res_file.write('Results\n\n')
    #add param values
    res_file.write(f'initial_training: {train}\n')
    res_file.write(f'quantization: {quantize}\n')
    res_file.write(f'few_shot_learning; {few_shot_learning}\n')
    res_file.write(f'num_loops: {bootstrap_num}\n')
    res_file.write(f'compare_inf_times_and_energy: {compare_inf_times_and_energy}\n')
    res_file.write(f'debug: {debug}\n')
    res_file.write(f'only_certain_iter: {only_certain_iter}\n')
    res_file.write(f'run_cpu: {run_cpu}\n')
    res_file.write(f'learn_comp_vals: {learn_comp_vals}\n')
    res_file.write(f'chanInds: {chanInds}\n')
    res_file.write(f'max_epochs_tuple: {max_epochs_tuple}\n\n')
    res_file.close()
    paths.update({'res_file_path':res_file_path})
        
    # create storage
    final_metrics_hist                       = np.zeros([3*2, bootstrap_num])
    few_shot_ind_driver_prelim_metrics_hist  = np.zeros([3, 11-num_train_drivers, bootstrap_num])
    few_shot_ind_driver_final_metrics_hist   = np.zeros([3, 11-num_train_drivers, bootstrap_num])
    ind_driver_convergence_epochs_hist       = np.zeros([11-num_train_drivers, bootstrap_num])
    ind_driver_inf_times_and_energy_hist     = np.zeros([6,11-num_train_drivers, bootstrap_num])
     
    #main loop
    for loop_iter in range(bootstrap_num):
        res_file = open(res_file_path, 'a')
        res_file.write(f'Iteration: {loop_iter}\n')
        res_file.write('\n')
        res_file.close()
        if len(only_certain_iter) > 0:
            if (loop_iter+1) not in only_certain_iter:
                print(f'skipping loop {loop_iter + 1}')
                continue
        loop_iter_str = str(loop_iter+1)
        results_path = os.path.join(paths.get('results_path_init'), f'{loop_iter_str}')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        models_path = os.path.join(results_path, 'models')
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        paths.update({'results_path':results_path,
                      'models_path':models_path})
        params.update({'bootstrap_idx':loop_iter})
        
        #train standard keras CNN on exp3 data
        if train:
            #update params
            stage      = 'train'
            if debug:
                max_epochs = 5
            else:
                max_epochs = max_epochs_tuple[0]
            params.update({'stage':stage,'max_epochs':max_epochs})
            
            #update paths
            save_fig_path = os.path.join(paths.get('results_path'), 'Training')
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)
            
            #import
            from tools_exp3 import supervised_learning
            from tools_exp3 import plot_metrics_from_hist
            
            #train
            trained_net, train_hist, eval_score_1 = supervised_learning(paths, params)
                
            #record final score
            res_file = open(res_file_path, 'a')
            res_file.write(f'Initial trained model final scores: acc = {eval_score_1[1]}, TPR = {eval_score_1[2]}, TNR = {eval_score_1[3]}\n\n')
            res_file.close()
            
            #plot results
            plot_metrics_from_hist(train_hist, save_fig_path)
            
            #save results
            final_metrics_hist[0,loop_iter] = eval_score_1[1] #train_hist.history['accuracy'][-1]
            final_metrics_hist[1,loop_iter] = eval_score_1[2] #train_hist.history['TPR'][-1]
            final_metrics_hist[2,loop_iter] = eval_score_1[3] #train_hist.history['TNR'][-1]
            
        if quantize:
            #update params
            stage = "quantize"
            if debug:
                max_epochs = 5
            else:
                max_epochs = max_epochs_tuple[1]
            params.update({'stage':stage,'max_epochs':max_epochs})
            
            #update paths
            save_fig_path = os.path.join(paths.get('results_path'), 'Quantize_Aware_Training')
            if not os.path.exists(save_fig_path):
                os.mkdir(save_fig_path)
                
            #quantize
            from tools_exp3 import quantize_model
            quant_ints = [8,8,1]
            net_quantized = quantize_model(trained_net, params, quant_ints)
            
            #quantization aware retraining
            from tools_exp3 import supervised_learning
            from tools_exp3 import plot_metrics_from_hist
            
            #train
            net_quantized_tuned, train_hist_quant, eval_score_2 = supervised_learning(paths, params, net_quantized)
            
            #record final score
            res_file = open(res_file_path, 'a')
            res_file.write(f'Quantized model final scores: acc = {eval_score_2[1]}, TPR = {eval_score_2[2]}, TNR = {eval_score_2[3]}\n\n')
            res_file.close()
            
            #plot results
            plot_metrics_from_hist(train_hist_quant, save_fig_path)
            
            #save results
            final_metrics_hist[3,loop_iter] = eval_score_2[1] #train_hist_quant.history['accuracy'][-1]
            final_metrics_hist[4,loop_iter] = eval_score_2[2] #train_hist_quant.history['TPR'][-1]
            final_metrics_hist[5,loop_iter] = eval_score_2[3] #train_hist_quant.history['TNR'][-1]
            
        if few_shot_learning:
            
            #import stuff
            import numpy as np
            from cnn2snn import convert
            from tools_exp3 import load_data
            from tools_exp3 import plot_metrics, estimate_spikes, estimate_num_neurons
            from tools_exp3 import edge_learning, get_prelim_performance
            from akida import AkidaUnsupervised
            from akida import FullyConnected
            
            #flags
            estimate_neurons = 0
            
            #update params
            stage = 'few_shot_learning'
            if debug:
                max_epochs = 2
            else:
                max_epochs = max_epochs_tuple[-1]
                    
            params.update({'stage':stage,'max_epochs':max_epochs})
            
            #convert model
            net_akida = convert(net_quantized_tuned)
                
            #load dataset
            datasets, dataset_lengths = load_data(paths, params)
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
                            
                    #batch and shuffle dataset
                    if params.get('batch_size') == -1:
                        batch_size = dataset_length
                    else:
                        batch_size = params.get('batch_size')
                    dataset = dataset.shuffle(params.get('shuffle_buffer_size')).batch(batch_size)
                
                    #prep model
                    #drop last layer
                    print('adding new layer..')
                    net_akida.pop_layer()
                    
                    #estimate spikes
                    spike_est = estimate_spikes(net_akida, dataset)
                    
                    #estimate neurons if flag
                    if estimate_neurons:
                        neurons_per_class = [
                        2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
                        100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000]
                        
                        weights_mults = [0.1, 0.5, 1, 1.2, 1.5, 2, 2.5]
                        
                        initial_plasticity = 0.5
                        min_plasticity = 0.05
                        plasticity_decay = 0.01
                        
                        #estimate neurons
                        est_neurons_params = {'dataset_length':dataset_length, 'neurons_per_class':neurons_per_class,
                                  'weights_mults':weights_mults,
                                  'save_path':results_path,
                                  'spike_est':spike_est,
                                  'initial_plasticity':initial_plasticity,
                                  'min_plasticity':min_plasticity,
                                  'plasticity_decay':plasticity_decay}
                        estimate_num_neurons(net_akida, dataset, est_neurons_params)
                        
                    #define edge learning params
                    weight_mult = 1.2
                    num_weights = int(np.ceil(weight_mult * spike_est))
                    num_neurons_per_class = 1000 #500
                        
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
                          
                    #update path
                    few_shot_results_path = os.path.join(results_path, 'few_shot_learning')
                    if not os.path.exists(few_shot_results_path):
                        os.mkdir(few_shot_results_path)
                    save_fig_path = os.path.join(few_shot_results_path, f'driver_{learn_iter + 1}')
                    if not os.path.exists(save_fig_path):
                        os.mkdir(save_fig_path)
                    
                    #get preliminary performance on held out data
                    prelim_metrics = get_prelim_performance(net_akida, dataset, paths)
                    #save prelim metrics
                    prelim_met_file = open(os.path.join(save_fig_path, 'prelim_metrics.txt'), 'w')
                    prelim_met_file.write(f'Acc: {prelim_metrics[0]}\n')
                    prelim_met_file.write(f'TPR: {prelim_metrics[1]}\n')
                    prelim_met_file.write(f'TNR: {prelim_metrics[2]}\n')
                    prelim_met_file.close()
                    
                    #save prelim performance
                    few_shot_ind_driver_prelim_metrics_hist[0,learn_iter,loop_iter] = prelim_metrics[0]
                    few_shot_ind_driver_prelim_metrics_hist[1,learn_iter,loop_iter] = prelim_metrics[1]
                    few_shot_ind_driver_prelim_metrics_hist[2,learn_iter,loop_iter] = prelim_metrics[2]
                    
                    #edge learn
                    net_akida, metrics = edge_learning(net_akida, dataset, params, paths)
                    #save edge learning metrics
                    metrics_npy = np.array(metrics)
                    np.save(os.path.join(save_fig_path, 'metrics.npy'), metrics_npy)
                    
                    #plot
                    plot_metrics(metrics, save_fig_path)
                    
                    #save results
                    few_shot_ind_driver_final_metrics_hist[0,learn_iter,loop_iter] = metrics[0][-1]
                    few_shot_ind_driver_final_metrics_hist[1,learn_iter,loop_iter] = metrics[1][-1]
                    few_shot_ind_driver_final_metrics_hist[2,learn_iter,loop_iter] = metrics[2][-1]
                    
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
                        
                    #record if converged
                    if epoch_converged > -1:
                        
                        #get inference time and energy
                        if compare_inf_times_and_energy:
                            from tools_exp3 import get_inf_times_and_energy
                            
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
    np.save(os.path.join(paths.get('results_path_init'), 'Ind_Model_prelim_metrics.npy'), few_shot_ind_driver_prelim_metrics_hist)
    np.save(os.path.join(paths.get('results_path_init'), 'Ind_Model_final_metrics.npy'), few_shot_ind_driver_final_metrics_hist)
    np.save(os.path.join(paths.get('results_path_init'), 'Ind_Model_Convergence_Epochs.npy'), ind_driver_convergence_epochs_hist)
    np.save(os.path.join(paths.get('results_path_init'), 'Ind_Model_inf_times_and_energies.npy'), ind_driver_inf_times_and_energy_hist)
    np.save(os.path.join(paths.get('results_path_init'), 'Group_Model_final_metrics.npy'), final_metrics_hist)
            
            
            
    #compute averages and standard deviations
    #calculate and record final metrics
    grand_ave = np.mean(final_metrics_hist, axis = 1)
    std_dev = np.std(final_metrics_hist, axis = 1)
    few_shot_ind_driver_metrics_shape     = np.shape(few_shot_ind_driver_prelim_metrics_hist)
    grand_ave_few_shot_ind_driver_prelim  = np.mean(np.reshape(few_shot_ind_driver_prelim_metrics_hist,
                                                               [few_shot_ind_driver_metrics_shape[0],
                                                                few_shot_ind_driver_metrics_shape[-2]*few_shot_ind_driver_metrics_shape[-1]]),
                                                    axis = -1)
    std_dev_few_shot_ind_driver_prelim    = np.std(np.reshape(few_shot_ind_driver_prelim_metrics_hist,
                                                               [few_shot_ind_driver_metrics_shape[0],
                                                                few_shot_ind_driver_metrics_shape[-2]*few_shot_ind_driver_metrics_shape[-1]]),
                                                    axis = -1)
    grand_ave_few_shot_ind_driver_final   = np.mean(np.reshape(few_shot_ind_driver_final_metrics_hist,
                                                               [few_shot_ind_driver_metrics_shape[0],
                                                                few_shot_ind_driver_metrics_shape[-2]*few_shot_ind_driver_metrics_shape[-1]]),
                                                    axis = -1)
    std_dev_few_shot_ind_driver_final     = np.std(np.reshape(few_shot_ind_driver_final_metrics_hist,
                                                               [few_shot_ind_driver_metrics_shape[0],
                                                                few_shot_ind_driver_metrics_shape[-2]*few_shot_ind_driver_metrics_shape[-1]]),
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
    res_file.write('Exp 2 Training:\n')
    res_file.write(f'Accuracy: {grand_ave[0]} ({std_dev[0]}), ')
    res_file.write(f'TPR: {grand_ave[1]} ({std_dev[1]}), ')
    res_file.write(f'TNR: {grand_ave[2]} ({std_dev[2]})\n\n')
    res_file.write('Quantization Aware Retraining:\n')
    res_file.write(f'Accuracy: {grand_ave[3]} ({std_dev[3]}), ')
    res_file.write(f'TPR: {grand_ave[4]} ({std_dev[4]}), ')
    res_file.write(f'TNR: {grand_ave[5]} ({std_dev[5]})\n\n')
    res_file.write('Few Shot Learning, Ind Driver: Prelim:\n')
    res_file.write(f'Accuracy: {grand_ave_few_shot_ind_driver_prelim[0]} ({std_dev_few_shot_ind_driver_prelim[0]}), ')
    res_file.write(f'TPR: {grand_ave_few_shot_ind_driver_prelim[1]} ({std_dev_few_shot_ind_driver_prelim[1]}), ')
    res_file.write(f'TNR: {grand_ave_few_shot_ind_driver_prelim[2]} ({std_dev_few_shot_ind_driver_prelim[2]})\n\n')
    res_file.write('Few Shot Learning, Ind Driver: Final:\n')
    res_file.write(f'Accuracy: {grand_ave_few_shot_ind_driver_final[0]} ({std_dev_few_shot_ind_driver_final[0]}), ')
    res_file.write(f'TPR: {grand_ave_few_shot_ind_driver_final[1]} ({std_dev_few_shot_ind_driver_final[1]}), ')
    res_file.write(f'TNR: {grand_ave_few_shot_ind_driver_final[2]} ({std_dev_few_shot_ind_driver_final[2]})\n\n')
    res_file.write(f'Number of iterations not converged: Ind Model: {num_zero_ind_model}\n')
    res_file.write(f'Few Shot Learning, Ind Drivers 90% Convergence Epochs: {grand_ave_ind_driver_convergence_epochs} ({std_dev_ind_driver_convergence_epochs})\n')
    if compare_inf_times_and_energy:
        #record inf times and energy mean and std for all ind driver data
        res_file.write(f'Ind Driver single epoch inference times (s): akida chip - {ind_driver_inf_t_and_e_grnd_ave[0]} ({ind_driver_inf_t_and_e_std[0]}), CPU - {ind_driver_inf_t_and_e_grnd_ave[2]} ({ind_driver_inf_t_and_e_std[2]})\n')
        res_file.write(f'Ind Driver single epoch inference energy (mJ): akida chip - {ind_driver_inf_t_and_e_grnd_ave[1]} ({ind_driver_inf_t_and_e_std[1]})\n')
        res_file.write(f'Ind Driver single epoch inference energy (mJ): CPU (baseline) - {ind_driver_inf_t_and_e_grnd_ave[5]} ({ind_driver_inf_t_and_e_std[5]}), CPU (total) - {ind_driver_inf_t_and_e_grnd_ave[4]} ({ind_driver_inf_t_and_e_std[4]}), CPU (adjusted) - {ind_driver_inf_t_and_e_grnd_ave[3]} ({ind_driver_inf_t_and_e_std[3]})\n')
    res_file.close()
        

#run main function of script if running from top-level
if __name__=='__main__':
    import sys
    sys.exit(main())        