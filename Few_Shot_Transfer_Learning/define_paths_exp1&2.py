# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 09:25:40 2024

@author: nalmrb
"""

def main(use_exp1_data=0):
    import os
    
    #define paths
    exp1_train_data_path = r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\Exp1Data'
    exp2_train_data_path = r"/home/nalmrb/train/"
    exp2_test_data_path = r'/home/nalmrb/test/'
    load_baseline_model_path = r"C:\Users\nalmrb\Desktop\ExperimentFiles\akida_debug\Bootstrap_Results\Exp1_Training\keras_cnn_exp1.keras"
    paths = {'exp1_train_data_path':exp1_train_data_path,
             'exp2_train_data_path':exp2_train_data_path,
             'exp2_test_data_path':exp2_test_data_path,
             'load_baseline_model_path':load_baseline_model_path}
    
    if use_exp1_data:
        bootstrap_data_path = r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\exp1_by_driver'
        bootstrap_results_path = r'C:\Users\nalmrb\Desktop\ExperimentFiles\akida_debug\Exp1_Analysis_Results'
    else:
        bootstrap_data_path = r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\MasterMetaL\total'
        bootstrap_results_path = r'C:\Users\nalmrb\Desktop\ExperimentFiles\akida_debug\Exp2_Analysis_Results'
    
    paths.update({'bootstrap_results_path':bootstrap_results_path,
                  'bootstrap_data_path':bootstrap_data_path})
        
    return paths
    
    
#run main function of script if running from top-level
if __name__=='__main__':
    import sys
    sys.exit(main())