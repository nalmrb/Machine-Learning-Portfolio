#!/usr/bin/env python
# coding: utf-8

#TPR Machine Learning Notebook
#Nathan Lutes
#06/25/2019

# This notebook uses a python module called Keras, a wrapper of tensorflow. The main idea in version 1.0.4 is to use a concept called transfer learning. Transfer learning is using elements of a pre-trained model combined with training a new fully connected layer to improve performance and training time. The pre-trained models included with keras are trained on the Imagenet dataset, a massive, 1000 class dataset. We will use the trained convolutional layers from these expertly crafted models to detect the features in our dataset and then the fully connected neural network top layer will perform the classification. The particular model used in this notebook will be InceptionResNet_V2.

# Imports

import numpy as np
import pandas as pd
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# Constants
#filepath
train_path = r'MakeThisYourTrainFolderPath'
test_path = r'MakeThisYourTestfolderPath'
results_path = r'WhereYouWantYourTestResultsStored'
#define input shape
#This needs to be the same size as the image slices
sliceSize = 1000    #default
rgb_layer = 3
hid_neur = 1024
num_classes = 2    #default
datasetSize = 299    #adjust accordingly
trainSize = round(datasetSize * 0.8)
testSize = datasetSize - trainSize
batchSize = 100    #set this to be whatever
norm_image = 1.0/255.0

# Construct Model
base_model = InceptionResNetV2(input_shape = (sliceSize, sliceSize, rgb_layer), include_top=False, weights='imagenet')

# We need to add a pooling, fully connected layer and an output layer now

# We add the pooling layer to reduce the dimension of the output from the convolutional layers to an input which will work in the dense, fully connected layer. The fully connected layers are the same as your standard neural network with one hidden layer and one output layer. This way the model runs the images through all of the filters contained in the convolutional layers and then the pooling layer reduces the dimensions of the input to the dense layer by taking the average of the convolution layer output. This is a common idea in convolutional neural networks and helps to prevent over-fitting of the data. Note that a flatten layer which just converts the matrix output of the convolutional layers into a suitable vector input for the dense layers would have also been acceptable. 

x = base_model.output
x = GlobalAveragePooling2D()(x)  #add the pooling layer
x = Dense(hid_neur, activation = 'relu', bias = True)(x)  # dense layer
pred = Dense(num_classes, activation = 'softmax')(x)  #output layer
model1 = Model(inputs=base_model.input, outputs = pred)

# Note that in the above cell we created a new model based on the InceptionResNet_V2 model. This is so we have more control over the model and so we still have an untouched base model to create additional models with if we want to compare performance. We will use the new model for training and testing

# set all layers as trainable (this will overwrite the previous layers which is not always best, but for this project I believe this is necessary)

for layer in base_model.layers:
    layer.trainable = True

# #### Now we compile the model. We are using RMSprop with Nestorov momentum as our learning algorithm and categorical cross entropy as our objective function

model1.compile(optimizer = 'Nadam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Construct Data Generator

# Now we need to construct the data generator using tools from keras

Datagen = image.ImageDataGenerator(rescale = norm_image)  #rescale to normalize pixel values
train_gen = Datagen.flow_from_directory(directory = train_path,
                                        target_size = (sliceSize, sliceSize),
                                       class_mode = 'categorical',
                                       shuffle = True)

test_gen = Datagen.flow_from_directory(directory = test_path,
                                       target_size = (sliceSize, sliceSize),
                                      class_mode = 'categorical',
                                      shuffle = False)


# Begin training model
model1.fit_generator(train_gen, steps_per_epoch = trainSize/batchSize, epochs = 100)

# Test the model
# Now that the training performance looks sufficient, let's test the model on the test data set that we set aside and that the model has not seen before

test_gen.reset()
results = model1.predict_generator(test_gen, steps_per_epoch = testSize/batchSize)
pred_labs = np.argmax(results, axis = 1)
true_labs = test_gen.classes

# Determine accuracy
count = 0
for i in range(0,len(pred_labs)):
    if pred_labs[i] == true_labs[i]:
        count += 1

perc_corr = count/len(pred_labs)
perc_corr

#Store labels and filenames to csv
labels = dict((v,k) for k,v in test_gen.class_indices.items())
predictions = [labels[k] for k in pred_labs]
filenames = test_gen.filenames[0:27300]
results = pd.DataFrame({"Filename":filenames,
                       "Predictions":predictions})
results.to_csv(results_path, index = False)

