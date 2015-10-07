% Cut the vertical seams from the source image
%
% input
% -----
% img : 3-d array of the source image
% energyMap : 2-d array accumulated energy map of the image
% horizSeam : 1-d array of the vertical seams
% 
% output
% ------
% carved : 3-d array of the image with the seams removed
% eM : 2-d array of the accumulated energy map with the seams removed
function [carved, eM] = vertSeamCarve(img, energyMap, vertSeam)
    [dimY, dimX, dimD] = size(img);
    carved = zeros(dimY, dimX-1, dimD);
    eM = zeros(dimY, dimX-1);
    
    for y=1:dimY
        % cut the seam out of the image and the energy map
        carved(y,:,:) = img(y,[1:vertSeam(y)-1,vertSeam(y)+1:dimX],:);
        eM(y,:) = energyMap(y,[1:vertSeam(y)-1,vertSeam(y)+1:dimX]);
    end;