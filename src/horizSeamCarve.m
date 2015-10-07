% Cut the horizontal seams from the source image
%
% input
% -----
% img : 3-d array of the source image
% energyMap : 2-d array accumulated energy map of the image
% horizSeam : 1-d array of the horizontal seams
% 
% output
% ------
% carved : 3-d array of the image with the seams removed
% eM : 2-d array of the accumulated energy map with the seams removed
function [carved, eM] = horizSeamCarve(img, energyMap, horizSeam)
    [dimY, dimX, dimD] = size(img);
    carved = zeros(dimY-1, dimX, dimD);
    eM = zeros(dimY-1, dimX);
    
    for x=1:dimX
        % cut the seam out of the image and the energy map
        carved(:,x,:) = img([1:horizSeam(x)-1,horizSeam(x)+1:dimY],x,:);
        eM(:,x) = energyMap([1:horizSeam(x)-1,horizSeam(x)+1:dimY],x);
    end;