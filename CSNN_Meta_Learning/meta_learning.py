# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:58:06 2023

@author: nalmrb
"""

def main():
    
    # This script is the main meta-learning control script, all parameters and
    # settings should be defined here
    
    #imports
    import os
    import sys
    import Define_paths
    from tools_and_functions import save_dict
    from meta_learn_functions import meta_learn_train, meta_learn_test
    from meta_learn_functions import initialize, make_meta_learning_training_plots
    
    #set flags
    train_config = 2  #0: tfs, 1: load from previous run, 2: load from pretrain
    if train_config == 0:
        #set based on train config
        pretrain_LN = 0
        reinitialize = 1
    elif train_config == 1:
        #set based on train config
        pretrain_LN = 0
        reinitialize = 0
    else:
        #set flag
        pretrain_LN = 0
        reinitialize = 1
    do_training = 1
    do_testing = 0
    bootstrap = 0
    
    #get paths
    paths = Define_paths.main()
    
    #define parameters
    #define LN params
    conv_filters = [3,12]
    conv_kernel_size = [5,3]
    LN_decay_adaptation = 5
    LN_thr = 0.02
    LN_decay = 0.5
    LN_dampening_factor = 0.3  
    n_reg_LN = 75 #arbitrary, change later
    n_ada_LN = 0 #arbitrary, change later
    n_out_LN = 2
    Beta_LN = 1.1
    num_steps = 10
    output_version = 2
    
    #define LSG/meta learning params
    n_reg_LSG = 100 #arbitrary, change later
    n_ada_LSG = 50 #arbitrary, change later
    n_out_LSG = n_reg_LN + n_ada_LN
    beta_LSG = 1.1
    learning_rate_outer = 5e-4
    learning_rate_inner = 5e-5
    window_size = 1
    num_data_per_driver = 4
    shuffle_buffer_size = 500
    num_updates = 1
    epochs_train = 3
    epochs_test = 50
    
    #save as dict
    LN_params = {'conv_filters':conv_filters,
             'conv_kernel_size':conv_kernel_size,
             'LN_decay_adaptation':LN_decay_adaptation,
             'LN_thr':LN_thr,
             'LN_decay':LN_decay,
             'LN_dampening_factor':LN_dampening_factor,
             'n_reg_LN':n_reg_LN,
             'n_ada_LN':n_ada_LN,
             'n_out_LN':n_out_LN,
             'Beta_LN':Beta_LN,
             'num_steps':num_steps,
             'output_version':output_version}
    
    #save as dict
    MetaL_params = {'n_reg_LSG':n_reg_LSG,
                'n_ada_LSG':n_ada_LSG,
                'n_out_LSG':n_out_LSG,
                'beta_LSG':beta_LSG,
                'learning_rate_outer':learning_rate_outer,
                'learning_rate_inner':learning_rate_inner,
                'window_size':window_size,
                'num_data_per_driver':num_data_per_driver,
                'shuffle_buffer_size':shuffle_buffer_size,
                'num_updates':num_updates,
                'epochs_train':epochs_train,
                'epochs_test':epochs_test,
                'train_config':train_config}
    
    #combine dicts into yet another dict
    params = {'LN_params':LN_params, 'MetaL_params':MetaL_params}
    
    if pretrain_LN == 1:
        from meta_learn_functions import Pretrain_LN
        
        #define additional pretrain params
        epochs_pretrain = 5000
        learning_rate_pretrain = 5e-5
        batch_size_pretrain = 448
        
        pretrain_params = {'epochs_pretrain':epochs_pretrain,
                           'learning_rate':learning_rate_pretrain,
                           'batch_size':batch_size_pretrain}
        
        #save
        save_dict(pretrain_params, paths.get('LN_pretrain_store_path'))
        
        #update params
        params.update({'pretrain_params':pretrain_params})
        
        #pretrain LN
        succeeded = Pretrain_LN(paths, params)
        
        #if didn't succeed, bomb script
        if succeeded == 0:
            sys.exit('Pretrain Failed.')
    
    if bootstrap == 0:
        #define store path
        base_store_path = paths.get('base_store_path')
        if train_config != 0:
            store_path = os.path.join(base_store_path,'pretrain_standard')
            paths.update({'store_path':store_path})
        else:
            store_path = os.path.join(base_store_path,'tfs_standard')
            paths.update({'store_path':store_path})
        
        #intialize
        paths = initialize(paths, params, train_config, reinitialize)
        
        #save
        save_dict(os.path.join(store_path, 'LN_params.txt'), LN_params)
        save_dict(os.path.join(store_path, 'MetaL_params.txt'), MetaL_params)
                
        #training part
        if do_training:
            #define data paths
            train_drivers_path = paths.get('train_drivers_data')
            train_drivers = os.listdir(train_drivers_path)
            data_paths = []
            for driver in train_drivers:
                data_paths.append(os.path.join(train_drivers_path, driver))
            
            #train
            meta_learn_train(data_paths, paths, params)
            
            #make figures
            make_meta_learning_training_plots(paths)
            
        if do_testing:
            #define data paths
            test_drivers_path = paths.get('test_drivers_data')
            test_drivers = os.listdir(test_drivers_path)
            data_paths = []
            for driver in test_drivers:
                data_paths.append(os.path.join(test_drivers_path, driver))
            
            #test
            meta_learn_test(data_paths, paths, params)
        
    


#run main function of script if running from top-level
if __name__ == '__main__':
    import sys
    sys.exit(main())