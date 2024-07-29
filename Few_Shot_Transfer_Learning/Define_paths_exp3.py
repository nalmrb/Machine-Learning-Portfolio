# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:53:00 2023

@author: nalmrb
"""

def main():
    import os
    
    data_path = r'C:\Users\nalmrb\Desktop\ExperimentFiles\Exp3_Python_data'
    results_path = r'C:\Users\nalmrb\Desktop\ExperimentFiles\exp3_debug\akida'
    load_baseline_model_path = r"C:\Users\nalmrb\Desktop\ExperimentFiles\akida_debug\Bootstrap_Results\Exp1_Training\keras_cnn_exp1.keras"
        
    paths = {'data_path':data_path, 'results_path_init':results_path,
             'load_baseline_model_path':load_baseline_model_path}
    return paths
        
    
#run main function of script if running from top-level
if __name__=='__main__':
    import sys
    sys.exit(main())