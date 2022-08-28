

clc
close all 
clear all

[filename, pathname] = uigetfile({'*.*';'*.bmp';'*.jpg';'*.gif'}, 'Pick a Leaf Image File');
I = imread([pathname,filename]);
newImage=imread(fullfile(pathname,filename));
I = imresize(I,[256,256]);
%figure, imshow(I); title('Query Leaf Image');

% Enhance Contrast
I = imadjust(I,stretchlim(I));
figure, imshow(I);title('Contrast Enhanced');

% Otsu Segmentation
I_Otsu = im2bw(I,graythresh(I));
% Conversion to HSI
I_HIS = rgb2hsi(I);


cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end



figure, subplot(3,1,1);imshow(segmented_images{1});title('Cluster 1'); subplot(3,1,2);imshow(segmented_images{2});title('Cluster 2');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));

% Feature Extraction
x = inputdlg('Enter the cluster no. containing the ROI only:');
i = str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};

% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end
%figure, imshow(img); title('Gray Scale Image');

% Evaluate the disease affected area
black = im2bw(seg_img,graythresh(seg_img));
%figure, imshow(black);title('Black & White Image');
m = size(seg_img,1);
n = size(seg_img,2);

zero_image = zeros(m,n); 
%G = imoverlay(zero_image,seg_img,[1 0 0]);

cc = bwconncomp(seg_img,6);
diseasedata = regionprops(cc,'basic');
A1 = diseasedata.Area;
sprintf('Area of the disease affected region is : %g%',A1);

I_black = im2bw(I,graythresh(I));
kk = bwconncomp(I,6);
leafdata = regionprops(kk,'basic');
A2 = leafdata.Area;
sprintf(' Total leaf area is : %g%',A2);

%Affected_Area = 1-(A1/A2);
Affected_Area = (A1/A2);
if Affected_Area < 0.1
    Affected_Area = Affected_Area+0.15;
end
sprintf('Affected Area is: %g%%',(Affected_Area*100))

% Create the Gray Level Cooccurance Matrices (GLCMs)
glcms = graycomatrix(img);

% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
%%
% Load All The Features
load('Training_Data.mat')

% Put the test features into variable 'test'
test = feat_disease;


% (1) Initialization, Assumption 
n  =  2;           % Dimension of search space
S  = 60;           % Number of bacteria in the colony
Nc = 25;           % Number of chemotactic steps 
Ns =  4;           % Number of swim steps 
Nre=  4;           % Number of reproductive steps 
Ned=  2;           % Number of elimination and dispersal steps
Sr =S/2;           % The number of bacteria reproductions (splits) per generation 
Ped=0.5;          % The probability that each bacteria will be eliminated/dispersed 
c(:,1)=0.05*ones(S,1);   % the run length unit (the size of the step taken in each run or tumble)
% Initial positions
for m=1:S                    % the initital posistions 
    B(1,:,1,1,1)= 10*rand(S,1)';
    B(2,:,1,1,1)= 10*rand(S,1)';
end  
%% Loops
% (2) Elimination-dispersal loop
for l = 1:Ned
    % (3) Reproduction loop
    for k = 1:Nre    
        % (4) Chemotaxis (swim/tumble) loop
        for j=1:Nc
            % (4.1) Chemotatic step
            for i=1:S 
                % (4.2) Fitness function
                J(i,j,k,l) = fitnessBFO(B(:,i,j,k,l));
                % (4.3) Jlast
                Jlast=J(i,j,k,l);
                % (4.4) Tumble
                Delta(:,i) = unifrnd(-1,1,n,1); 
                % (4.5) Move
                B(:,i,j+1,k,l)=B(:,i,j,k,l)+c(i,k)*Delta(:,i)/sqrt(Delta(:,i)'*Delta(:,i));
                % (4.6) New fitness function
                J(i,j+1,k,l)=fitnessBFO(B(:,i,j+1,k,l));
                % (4.7) Swimming
                m=0; % counter for swim length
                while m < Ns 
                    m=m+1;
                     if J(i,j+1,k,l)<Jlast  
                        Jlast=J(i,j+1,k,l);    
                        B(:,i,j+1,k,l)=B(:,i,j+1,k,l)+c(i,k)*Delta(:,i)/sqrt(Delta(:,i)'*Delta(:,i)) ;  
                        J(i,j+1,k,l)=fitnessBFO(B(:,i,j+1,k,l));  
                     else       
                        m=Ns;     
                     end 
                end
                J(i,j,k,l)=Jlast; %???
            end % (4.8) Next bacterium
            x = B(1,:,j,k,l);
            y = B(2,:,j,k,l);
            clf % clears figure 
                run rose_fungraph.m
                plot(x,y,'*','markers',6) % plots figure
                axis([-1.5 1.5 -1 3]), axis square
                xlabel('x'); ylabel('y')
                title('Bacterial Foraging Optimization'); grid on
                legend('Rosenbrock function','Bacteria')
                pause(.01)
                hold on
        end % (5) if j < Nc, chemotaxis
        % (6) Reproduction
        % (6.1) Health
        Jhealth=sum(J(:,:,k,l),2);      % Set the health of each of the S bacteria
        [Jhealth,sortind]=sort(Jhealth);% Sorts bacteria in order of ascending values
        B(:,:,1,k+1,l)=B(:,sortind,Nc+1,k,l); 
        c(:,k+1)=c(sortind,k);          % Keeps the chemotaxis parameters with each bacterium at the next generation
        % (6.2) Split the bacteria
        for i=1:Sr % Sr??
                B(:,i+Sr,1,k+1,l)=B(:,i,1,k+1,l); % The least fit do not reproduce, the most fit ones split into two identical copies  
                c(i+Sr,k+1)=c(i,k+1);                 
        end
    end % (7) Loop to go to the next reproductive step
    % (8) Elimination-dispersal
        for m=1:S 
            if  Ped>rand % % Generate random number 
                B(1,:,1,1,1)= 50*rand(S,1)';
                B(2,:,1,1,1)= .2*rand(S,1)';  
            else 
                B(:,m,1,1,l+1)=B(:,m,1,Nre+1,l); % Bacteria that are not dispersed
            end        
        end 
end
%% Results
           reproduction = J(:,1:Nc,Nre,Ned);
           [jlastreproduction,O] = min(reproduction,[],2);  % min cost function for each bacterial 
           [Y,I] = min(jlastreproduction);
           pbest = B(:,I,O(I,:),k,l);
           display('Best solution:')
           display(['x = ' mat2str(pbest(1),2)])
           display(['y = ' mat2str(pbest(2),2)])
           plot(pbest(1),pbest(2),'ro')
           hold off
           legend('Rosenbrock function','Bacteria','Best solution')

%% Load Images
rootFolder = fullfile('dataset');
%rootFolder = fullfile(outputFolder, 'dataset');
categories = {'Alternaria Alternata', 'Anthracnose','Bacterial Blight','Cercospora Leaf Spot','Healthy Leaves'};


imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


tbl = countEachLabel(imds)


minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)



% Find the first instance of an image for each category
basophil = find(imds.Labels == 'Alternaria Alternata', 1);
eosinophil = find(imds.Labels == 'Anthracnose', 1);
lymphocyte = find(imds.Labels == 'Bacterial Blight', 1);
monocyte = find(imds.Labels == 'Cercospora Leaf Spot', 1);
neutrophil = find(imds.Labels == 'Healthy Leaves', 1);


figure(1)
subplot(1,3,1);
imshow(readimage(imds,basophil))
title('Alternaria Alternata') 
subplot(1,3,2);
imshow(readimage(imds,eosinophil))
title('Anthracnose')
figure(2)
subplot(1,3,1);
imshow(readimage(imds,lymphocyte))
title('Bacterial Blight')
subplot(1,3,2);
imshow(readimage(imds,monocyte))
title('Cercospora Leaf Spot')
subplot(1,3,3);
imshow(readimage(imds,neutrophil))
title('Healthy Leaves')


% Load pretrained network
net = resnet50();


figure
plot(net)
title('First section of ResNet-50')
set(gca,'YLim',[150 170]);


% Inspect the first layer
net.Layers(1)

 

% Inspect the last layer
net.Layers(end)

% Number of class names for ImageNet classification task
numel(net.Layers(end).ClassNames)




[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

%% Pre-process Images For CNN

% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');



% Get the network weights for the second convolutional layer
w1 = net.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5); 

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')


featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');



% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');



% Extract test features using the CNN
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2))
%%

% Display the mean accuracy
mean(diag(confMat))


ds = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');

% Extract image features using the CNN
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');
%%

% Make a prediction using the classifier
label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')




