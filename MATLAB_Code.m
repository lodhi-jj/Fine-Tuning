
%{
%% Load a pre-trained, deep, convolutional network
net = alexnet
layers = net.Layers;

%% Modify the network to use 
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer

%% Setup training-testing data
rootFolder = fullfile('E:\No need here\Datasets\Yawning');
imds = imageDatastore(fullfile(rootFolder,'Alexnet_227_227'), 'LabelSource','foldernames', 'IncludeSubfolders',true);
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

%% Re-train the network
opts = trainingOptions('sgdm','InitialLearnRate', 0.001, 'MaxEpochs', 20, 'MiniBatchSize', 64);
densenet = trainNetwork(trainingSet,layers,opts);

%% Measure Network Accuracy
predLabels = classify(densenet, testSet);
accuracy = mean(predLabels == testSet.Labels);
%}

%% Resizing of images
% rootFolder = fullfile('E:\No need here\Datasets\Yawning');
% imds = imageDatastore(fullfile(rootFolder,'Orig'), 'LabelSource','foldernames', 'IncludeSubfolders',true);
% for i = 1:size(imds.Files,1)
%     im = imread(cell2mat(imds.Files(i)));
%     im = imresize(im, [227 227]);
%     sp = split(imds.Files(i),'\');
%     if string(sp{end-1}) == 'Yawn'
%         cd ('E:\No need here\Datasets\Yawning\Alexnet_227_227\Yawn')
%         imwrite(im, sp{end});
%     elseif string(sp{end-1}) == 'Not_Yawn'
%         cd ('E:\No need here\Datasets\Yawning\Alexnet_227_227\Not_Yawn')
%         imwrite(im, sp{end});
%     end
% end

%% Testing on Camera
clear

camera = webcam; % Connect to the camera
nnet = trainnet;  % Load the neural net

while true   
    picture = camera.snapshot;              % Take a picture    
    picture = imresize(picture,[227,227]);  % Resize the picture

    label = classify(nnet, picture);        % Classify the picture
       
    image(picture);     % Show the picture
    title(char(label)); % Show the label
    drawnow;   
end