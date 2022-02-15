% Creates the vectors for the color transfer example
% This function requires that you manually add the val2017 dataset from the of the Coco Dataset to the Matlab path. 

n = 100;
rgbImage = imread('000000002149.jpg');
rgbImage = imresize(rgbImage,[n n]);
X1 = reshape(rgbImage,n^2,3);
%imshow(reshape(X1,n,n,3));

%%
rgbImage = imread('000000544306.jpg');
rgbImage = imresize(rgbImage,[n n]);
X2 = reshape(rgbImage,n^2,3);
%imshow(reshape(X2,n,n,3));

%%
rgbImage = imread('000000079408.jpg');
rgbImage = imresize(rgbImage,[n n]);
X3 = reshape(rgbImage,n^2,3);
%imshow(reshape(X3,n,n,3));

%%
rgbImage = imread('000000291490.jpg');
rgbImage = imcrop(rgbImage,[1.275100000000000e+02,71.510000000000000,3.159800000000000e+02,2.689800000000000e+02]);
rgbImage = imresize(rgbImage,[n n]);
X4 = reshape(rgbImage,n^2,3);
%imshow(reshape(X4,n,n,3));

%% 
X1 = double(X1)/256;
X2 = double(X2)/256;
X3 = double(X3)/256;
X4 = double(X4)/256;

save('pictures.mat','X1','X2','X3','X4','n')
