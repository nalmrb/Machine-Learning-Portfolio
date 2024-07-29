# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:43:05 2023

@author: nalmrb
"""

from glob import glob
import os
import numpy as np
import re

data_dir = r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\exp1_by_driver' #r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\MasterMetaL\total' # r'C:\Users\nalmrb\AppData\Roaming\MobaXterm\home\exp3_files\Exp3_Python_data'

#define different permuts
# driver_permuts = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                   [4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3],
#                   [7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 6],
#                   [10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                   [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1],
#                   [5, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4],
#                   [8, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7],
#                   [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#                   [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2],
#                   [6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5]]

driver_permuts = [[7,8,9,11,13,15,17,1,2,4,6],
                    [1,2,4,6,7,8,9,11,13,15,17],
                    [11,13,15,17,1,2,4,6,7,8,9],
                    [6,7,8,9,11,13,15,17,1,2,4],
                    [17,1,2,4,6,7,8,9,11,13,15],
                    [9,11,13,15,17,1,2,4,6,7,8],
                    [4,6,7,8,9,11,13,15,17,1,2],
                    [15,17,1,2,4,6,7,8,9,11,13],
                    [8,9,11,13,15,17,1,2,4,6,7],
                    [2,4,6,7,8,9,11,13,15,17,1]]

# driver_permuts = [[3,5,10,12,18,20,22,24,14,16,26],
#                      [12,18,20,22,24,14,16,26,3,5,10],
#                      [22,24,14,16,26,3,5,10,12,18,20],
#                      [16,26,3,5,10,12,18,20,22,24,14],
#                      [5,10,12,18,20,22,24,14,16,26,3],
#                      [18,20,22,24,14,16,26,3,5,10,12],
#                      [24,14,16,26,3,5,10,12,18,20,22],
#                      [26,3,5,10,12,18,20,22,24,14,16],
#                      [10,12,18,20,22,24,14,16,26,3,5],
#                      [20,22,24,14,16,26,3,5,10,12,18]]

totals = np.zeros([12,10])

for idx2 in range(10):

    #construct file list
    load_folder_nums_train = driver_permuts[idx2][0:8]
    load_folder_nums_test  = driver_permuts[idx2][8:]
    filePathList_train = []
    filePathList_test  = []
    folders = os.listdir(data_dir)
    for this_dir in folders:
        if os.path.isdir(os.path.join(data_dir, this_dir)):
            driver_dir_num = eval(re.findall(r'\d+', this_dir)[0])
            if driver_dir_num in load_folder_nums_train:
                filePathList_train.append(os.path.join(data_dir, this_dir))
            elif driver_dir_num in load_folder_nums_test:
                filePathList_test.append(os.path.join(data_dir, this_dir))
    
    filepathlistcounter = 0
    filePathLists = [filePathList_train] + filePathList_test
    for filePathList in filePathLists:
        filepathlistcounter += 1
        
        #counters
        go   = 0
        nogo = 0
        
        if type(filePathList) is not list:
            filePathList = [filePathList]
        
        for this_filePath in filePathList:
             
            #get directory contents
            fileList = glob(os.path.join(this_filePath, '*.npy'))
            
            # for file in fileList:
            #     if 'NoGo' in file:
            #         nogo += 1
            #     elif 'Go' in file:
            #         go += 1
            
            for file in fileList:
                data = np.load(file)
                if (996 - np.shape(data)[0]) < 0:
                    continue
                
                if '1-0' in file:
                    go += 1
                else:
                    nogo += 1
            
            idx1 = (filepathlistcounter-1) * 3
            totals[idx1,idx2]   = go + nogo
            totals[idx1+1,idx2] = go
            totals[idx1+2,idx2] = nogo
            

#get averages and standard deviations
totals_ave = np.mean(totals, axis = 1)
totals_std = np.std(totals, axis = 1)
        
        
         
