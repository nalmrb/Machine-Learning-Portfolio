""" This is the module that will take a microscope image, resize it if needed,
split it into user defined sub-sections, aid the user in the labeling process,
and then store the images in the correct folder.

GUI edition
"""

#Nathan Lutes
#Image split and classify module
#06/14/2019

#initialize
def init():

    #importz
    import glob
    import os
    import numpy as np
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import messagebox
    import skimage as ski
    import PIL
    from PIL import ImageTk
    
    #Global var declaration and initialization
    global number1, number2, blk, fileloc, img_name, outfile, blk_size
    global num_blks, num_imgs, blks, image_set, img_file, im_blk, im_main
    number1 = 0  #image index count
    number2 = 0  #block index count
    im_blk = None
    im_main = None
    
    #Constantz to eliminate 'magic numbers'
    rgb_layers = 3
    blk_size = 1000  #default blk_size
    # default paths used for easy debugging
#    outfile = r'C:\Users\nalutes\desktop\test_del_me'
    
    #init routine
    #get all dem imagez
    #Create GUI     
    root = tk.Tk()  #Create main window
    
    #file creation loopz
    cond_1 = False  #initialize
    cond_2 = False
    while cond_1 == False:
        # Have user select parent file
        try:
            messagebox.showinfo (message = 'Please select folder containing images')
            fileloc = filedialog.askdirectory(title = 'Please select file '
                                              'containing images')
        except:
            messagebox.showinfo(message = 'This file location is invalid')
        image_set = glob.glob(fileloc + '\\' + '*.jpg')
        num_imgs = len(image_set)
        if image_set:
            cond_1 = True
        else:
            messagebox.showwarning(message = 'There are no images in this file')
            
    while cond_2 == False:
        # Have user select output file
        try:
            messagebox.showinfo(message = 'Please select file for subfolder creation')
            outfile = filedialog.askdirectory(title = 'Please select file'
                                              'for subfolder creation')
        except:
            messagebox.showinfo(message = 'This file location is invalid')
            
        try:
            sample_path = outfile + '\\' + 'samples'
            trash_path = outfile + '\\' + 'discards'
            sample_abio = sample_path + '\\' + 'Abiotic'
            sample_bio = sample_path + '\\' + 'Biotic'
            trash_unk = trash_path + '\\' + 'Unknown'
            trash_disc = trash_path + '\\' + 'Discard'
            os.mkdir(sample_path)
            os.mkdir(trash_path)
            os.mkdir(sample_abio)
            os.mkdir(sample_bio)
            os.mkdir(trash_unk)
            os.mkdir(trash_disc)
            cond_2 = True
        except:
            messagebox.showwarning(message = 'Subfiles already exist.')
            cond_2 = True
            
    #Check if a record has been made and read it if possible
    try:
        with open(outfile + '\\' + 'record.txt', 'r') as myfile:
            record = myfile.read()
        #Create a list from the record file and remove any corresponding images
        #from the image set
        record_list = record.split('\n')
        for i in range(0, len(record_list)):
            if record_list[i] in image_set:
                image_set.remove(record_list[i])
    except IOError:
        #This means that there is no record file in this file location
        pass  #do nothing
                  
    #functionz
    def set_blk_size():
        #This function allows the user to set the block size
        global blk_size
        #get the input from the user, if it is valid, destroy the frame
        try:
            blk_size = int(user_input.get())
            fr_blk_size.destroy()
        except:
            #if the input is invalid, return to the default
            messagebox.showinfo(message = 'Entered dimensions invalid'
                                )
            blk_size = 1000
            
    def disp_img():
        # Function used to keep of track and display images to the screen
        global number1, number2, num_blks, num_imgs, img_file, im_blk, im_main
        global blk
        # For the first image
        if number1 == 0 and number2 == 0:
            process_image()
            #display first large image
            img_=PIL.Image.open(img_file)
            im_main = PIL.ImageTk.PhotoImage(img_.resize([500,500]))
            canvas1.create_image(2,2, image = im_main, anchor = 'nw')
            canvas1.update_idletasks()
            #display first block
            in_arr = np.array(blks[number2], dtype = np.uint8)
            blk = PIL.Image.fromarray(in_arr)
            im_blk = PIL.ImageTk.PhotoImage(blk.resize([500,500]))
            canvas2.create_image(2,2, image = im_blk, anchor = 'nw')
            canvas2.update_idletasks()
            # increase index
            number2 += 1
        else:
            #check if all the blocks have been displayed
            if number2 == num_blks:     #python starts at zero
                try:
                    #change displayed large image to next image in file
                    with open(outfile + '\\' + 'record.txt', 'a') as myfile:
                        myfile.write(img_file + '\n')
                    number1 += 1  #index to next image in list
                    process_image()
                    img_=PIL.Image.open(img_file)
                    im_main = PIL.ImageTk.PhotoImage(img_.resize([500, 500]))
                    canvas1.create_image(2,2, image = im_main, anchor = 'nw')
                    canvas1.update_idletasks()
                    number2 = 0  #reset block count
                except IndexError:
                    #display task complete
                    messagebox.showinfo(message =  'No more images in file')
                    Exit() #exit the program
            #unless program is terminated, display next block
            try:
                in_arr = np.array(blks[number2], dtype = np.uint8)
            except:
                messagebox.showinfo(message = 'no more blocks')
            blk = PIL.Image.fromarray(in_arr)
            im_blk = ImageTk.PhotoImage(blk.resize([500,500]))
            canvas2.create_image(2,2, image = im_blk, anchor = 'nw')
            canvas2.update_idletasks()
            # increase index
            number2 += 1
    
    def process_image():
        #This function transforms the image appropriately and slices it into
        #blocks
        global number1, img_name, blk_size, blks, num_blks, image_set, img_file
        # get file
        img_file = image_set[number1]
        # get image name
        base = img_file.split('\\')[-1]
        img_name = base.split('.')[0]
        #create image object
        img_read = ski.io.imread(img_file)
        
        #slightly resize image by transforming it to the largest dimensions
        #that will allow splitting into blocks if necessary
        size = img_read.shape
        if size[0]%blk_size != 0:
            new_x = blk_size * round(size[0]/blk_size)
        else:
            new_x = size[0] 
        if size[1]%blk_size != 0:
            new_y = blk_size * round(size[1]/blk_size)
        else:
            new_y = size[1]
            
        #transform image if necessary
        img_read = ski.transform.resize(img_read, (new_x,new_y,rgb_layers),
                                    mode = 'constant', preserve_range = True)
        #split image into blocks
        blks = ski.util.view_as_blocks(img_read, block_shape=(blk_size,blk_size
                                                              ,rgb_layers))
        arr_dim = blks.shape  #get the shape so I can index it in the next line
        num_blks = arr_dim[0] * arr_dim[1]
        #reshape array to work with image display command
        blks = blks.reshape(num_blks, blk_size, blk_size, rgb_layers)
    
    def Abio():
        """ This function assigns an image to the abiotic folder and is to be
        used with the SplitnClass module
        """
        global number2, blk
        filename = (sample_abio + '\\' + img_name + '_' +
        str(number2) + '.jpg'
        )
        blk.save(filename)
        disp_img()

    def Bio():
        """ This function assigns an image to the biotic folder and is to be
        used with the SplitnClass module
        """
        global number2, blk
        filename = (sample_bio + '\\' + img_name + '_' +
                    str(number2) + '.jpg'
        )
        blk.save(filename)
        disp_img()
        
    def Unk():
        """ This function assigns an image to the unknown folder and is to be
        used with the SplitnClass module
        """
        global number2, blk
        filename = (trash_unk + '\\' + img_name + '_' +
                    str(number2) + '.jpg'
        )
        blk.save(filename)
        disp_img()
        
    def Discard():
        """ This function assigns an image to the unknown folder and is to be
        used with the SplitnClass module
        """
        global number2, blk
        filename = (trash_disc + '\\' + img_name + '_' +
                    str(number2) + '.jpg'
        )
        blk.save(filename)
        disp_img()
        
    def Exit():
        """ This function exits the GUI window
        """
        root.destroy()
    
     #Ask user for slice size input
    fr_blk_size = tk.Tk()
    blk_label_1 = tk.Label(fr_blk_size,
                           text = 'Set slice dimension? Default: 1000')
    blk_label_1.pack()
    user_input = tk.Entry(fr_blk_size)
    user_input.pack()
    yes_button = tk.Button(fr_blk_size, text = 'Yes', command = set_blk_size)
    yes_button.pack(side = 'left')
    no_button = tk.Button(fr_blk_size, text = 'No',
                          command = fr_blk_size.destroy)
    no_button.pack(side = 'right')
    fr_blk_size.wait_window()
    
    #Create auxilliary windows
    frame1 = tk.Toplevel(master = root)
    frame2 = tk.Toplevel(master = root)
    #populate gui with widgets
    canvas1 = tk.Canvas(frame2, image = None, width = 500, height = 500)
    canvas1.pack(side = 'left', fill = 'both', expand = 'True')
    canvas2 = tk.Canvas(frame1, image = None, width = 500, height = 500)
    canvas2.pack()
    #populate gui with buttons
    abio = tk.Button(master = root, text = 'Abiotic', command = Abio)
    abio.pack()
    bio = tk.Button(master = root, text = 'Biotic', command = Bio)
    bio.pack()
    unk = tk.Button(master = root, text = 'Unknown', command = Unk)
    unk.pack()
    disc = tk.Button(master = root, text = '"discard"', command = Discard)
    disc.pack()
    Exit1 = tk.Button(master = root, text = 'Close', command = Exit)
    Exit1.pack(side = 'bottom')
    
    #Run functions to display first image
    disp_img()    #run for every image change
    root.mainloop()  #start the mainloop
init()
