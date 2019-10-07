%Nathan Lutes
%ML program 7/16/2019

%This program is designed to build a predictive model for the life
%detection project by using transfer learning concepts. A
%pre-existing CNN will be modified to fit our goals

%%
%Load pre-existing model
%Navigate to folder containing stored network
Check = false;
if Check == true
    try
        load('FCNN8sTPR.mat')
        disp('Network loaded as net1')
    catch
        disp("Could not find Network or it doesn't exist")
    end
else
    pass
end

%%
%The images are different sizes, so they all need to be resized to be
%consistent
%define size
numrows = 500;
numcols = 500;
%if size of images is too large, try making the above values smaller
%However, I'm not sure if that will cause resolution issues
%Resize images
InputSize = [numrows numcols];
%%
%Load in the project data set
[imdsTrain, imdsTest, pxdsTrain, pxdsTest, imds, pxds] = LoadAndPrepareData(gTruth, InputSize);

%% 
%Analyze data statistics
classes = pxds.ClassNames;
labelIDs = 1:length(classes);
tbl = countEachLabel(pxds);
frequency = tbl.PixelCount/tbl.ImagePixelCount;
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%This creates a neat looking bargraph showing the frequency of each class

%% optional
%display an overlayed image from one of the new datastores
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pxdsTrain.ReadFcn = @(loc)imresize(imread(loc),InputSize);
ex_im = readimage(imdsTrain, 1);
ex_pll = readimage(pxdsTrain, 1);
B = labeloverlay(ex_im, ex_pll);
figure()
imshow(B)

%%
%set training options for the network
options = trainingOptions('adam',...
    'InitialLearnRate', .005, ...
    'MaxEpochs', 50, ...
    'ExecutionEnvironment', 'auto',...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'CheckpointPath', tempdir, ...
    'Verbose', true,...
    'VerboseFrequency', 1);

%%
%create and train the FCN network (if it doesn't exist already)
if exist('net1', 'var') == 1 && check == true
    %Further train an existing network
    net1 = networkTraining(net1, options);
else
    %new network
    %change properties inside of functions
    lgraph = createNewNetwork(pxds,tbl, InputSize);
    net1 = networkTraining(lgraph, options, imdsTrain, pxdsTrain, InputSize);
end

%%
%Test the network on a single testing image
imdsTest.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pxdsTest.ReadFcn = @(loc)imresize(imread(loc),InputSize);
picNum = 6;
test_img = readimage(imdsTest, picNum);
test_res = semanticseg(test_img, net1);
%display the results
tres_disp = labeloverlay(test_img, test_res, 'Transparency', 0.4);
imshow(tres_disp)
title('HAL')
%compare with 'ground truth'
figure()
g_truth = readimage(pxdsTest, picNum);
B2 = labeloverlay(test_img, g_truth);
imshow(B2)
title('Ground Truth')

%show the results overlap
%"intersection over union"
iou = jaccard(test_res, g_truth);
table(classes, iou)
%this shows a table of numbers corresponding to classes. This shows you the
%level of agreement the resulting image has with the ground truth image. A
%perfect match would have numbers of 1 for every class

%%
%Evaluate the network for the whole test set
imdsTest.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pxdsTest.ReadFcn = @(loc)imresize(imread(loc),InputSize);
outFolder = 'C:\Users\CT-User\Desktop';

for i = 1:length(imdsTest.Files)
    nxt = input('Press enter to see next image: ');
    test_img = readimage(imdsTest, i);
    test_res = semanticseg(test_img, net1);
    %display the results
    tres_disp = labeloverlay(test_img, test_res, 'Transparency', 0.4);
    imshow(tres_disp)
    title('HAL')
    %compare with 'ground truth'
    figure()
    g_truth = readimage(pxdsTest, i);
    B2 = labeloverlay(test_img, g_truth);
    imshow(B2)
    title('Ground Truth')

    %show the results overlap
    %"intersection over union"
    iou = jaccard(test_res, g_truth);
    table(classes, iou)
end

%%
%Compute confusion matrix and metrics for each class, each image and the
%entire dataset and store this information
%Calculate the metrics of the test results
testResults = semanticseg(imdsTest, net1);
metrics = evaluateSemanticSegmentation(testResults, pxdsTest);
%display some of the metrics results
metrics.ClassMetrics
metrics.NormalizedConfusionMatrix

%%
%optional save the network
save('FCNN8sTPR.mat', 'net1')

%%
function [imdsTrain, imdsTest, pxdsTrain, pxdsTest, imds, pxds] = LoadAndPrepareData(gTruth, InputSize)
%export labels from app
try
    [imds, pxds] = pixelLabelTrainingData(gTruth);
catch
    disp('Export from the Image Labeler app first')
    return
end

imds.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pxds.ReadFcn = @(loc)imresize(imread(loc),InputSize);

%split into training and testing sets 70/30 split
classes = pxds.ClassNames;
labelIDs = 1:length(classes);
%get the number of files in the dataset
numFiles = numel(imds.Files);   %a vector the size of the database
rndmIdx = randperm(numFiles);   %randomize the order of the images

%Get indices for training set
numTrain = round(0.7 * numFiles);
trainingIdx = rndmIdx(1:numTrain);
%Get indices for testing set
testIdx = rndmIdx(numTrain+1:end);

%Create individual datastores for partitioned sets
imdsTrain = imageDatastore(imds.Files(trainingIdx));
imdsTest = imageDatastore(imds.Files(testIdx));

%Create pixel label datastores
trainLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end
%%
function lgraph = createNewNetwork(pxds,tbl,InputSize)
classes = pxds.ClassNames;
numClasses = numel(classes);    %2
lgraph = fcnLayers(InputSize, numClasses);   %create the actual network

%Balance classes using class weighting
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

%specify class weights using a pixelClassificationLayer
pxlayer = pixelClassificationLayer('Name', 'labels',...
    'Classes', classes, 'ClassWeights', classWeights);

%remove the previous pixel classification layer from the network
%and add the newly created one
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxlayer);
lgraph = connectLayers(lgraph, 'softmax', 'labels');
%This section is very important because our images have very imbalanced
%classes (ie the ratio of non biotic pixels to biotic pixels is very high)
%As such, our network will need a strong penalty for missclassifying biotic
%pixels as abiotic in order to prevent it from over - generalizing to the
%abiotic label
end

%%
function net1 = networkTraining(net, options, imdsTrain, pxdsTrain, InputSize)
%set data augmentation
%This is important to improve the generalization capabilities of the
%network. It also helps accuracy when using a relatively small data set

augmenter = imageDataAugmenter('RandXReflection', true, ...
    'RandXTranslation', [-10, 10], 'RandYTranslation', [-10,10]);

%Start Training
%create generator using the training data and the data augmenter
%create training datastore
imdsTrain.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pxdsTrain.ReadFcn = @(loc)imresize(imread(loc),InputSize);
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'DataAugmentation', augmenter);

[net1, ~] = trainNetwork(pximds, net, options);
end
