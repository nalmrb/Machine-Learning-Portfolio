#Nathan Lutes
#Train/Test Creator
#06/25/2019

""" This program splits data that has been presorted into subfolders and
further sorts it into training and testing sets according to the prespecified
sorting ratio. This is so the flow from directory tool can be used with keras.
This program assumes all data are images of the jpeg data type
"""

#importz
import glob
import os
import random
import shutil
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

#Constantz
train_ratio = 0.8
test_ratio = 1 - train_ratio
    
#main loop

def main():
    
    #Get the folder from the user
    root = tk.Tk()
    messagebox.showinfo(message = 'select the folder containing the data'
                           ' subdirectories.')
    filepath = filedialog.askdirectory(title = 'select the folder containing '
                                       'the data subdirectories')
    root.withdraw()
    
    #First Create folders
    #I do this step first to stop the program if the folders already exist
    train_path = filepath + '\\' + 'train'
    test_path = filepath + '\\' + 'test'
    try:
        os.mkdir(train_path)
        os.mkdir(test_path)
    except FileExistsError:
        print('Train and Test files already exist, delete the old files and '
              'run the program again')
        sys.exit()
    
    #get list of folders from inside the filepath
    dir_list = [fname for fname in os.listdir(filepath) if fname != 'train'
                and fname != 'test' and fname != 'record.txt']
    print(dir_list)
    
    for i in dir_list:
        dir_list2 = [fname for fname in os.listdir(filepath + '\\' + i) if i == 'samples']
    print(dir_list2)
        
    #get files inside of samples file 
    content_list = [glob.glob(filepath + '\\' + 'samples' + '\\' + dir_name + '\\' + '*.jpg')
    for dir_name in dir_list2]  # a list of lists
    
    # get lengths of each list
    clist_lengths = [len(sublist) for sublist in content_list]
    print(clist_lengths)
    # find the smallest number in list
    #This is an important step so that the classes are balanced
    min_value = min(clist_lengths)
    #set the training and testing sizes
    train_size = int(min_value * train_ratio)
    test_size = int(min_value - train_size)
    
    #shuffle the contents of each list
    for sublist in content_list:
        random.shuffle(sublist)
    
    # now we need to save the respective amount of images into each folder
    #create the same class folders in the new train/test directories
    for subdir in dir_list2:
        try:
            os.mkdir(train_path + '\\' + subdir)
            os.mkdir(test_path + '\\' + subdir)
        except FileExistsError:
            pass  #if files already exist, do nothing
    
    #save appropriate amount of files into each subdirectory
    for i,subdir in enumerate(dir_list2):  #for every directory in the dir list
        #slice train and test sets from shuffled list
        train_set = content_list[i][0:train_size + 1]
        test_set = content_list[i][len(train_set):len(train_set) + test_size + 1]
        #copy the files in train_set to the train folder
        for file in train_set:
            shutil.copy(file, train_path + '\\' + subdir)
        #copy the files in test_set to the test folder
        for file in test_set:
            shutil.copy(file, test_path + '\\' + subdir)
            
main()