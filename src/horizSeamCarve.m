% Cut the horizontal seams from the source image
%
% input
% -----
% img : 3-d array of the source image
% horizSeam : 1-d array of the horizontal seams
% 
% output
% ------
% carved : 3-d array of the image with the seams removed
function [carved, eM] = horizSeamCarve(img, horizSeam)
    [dimY, dimX, dimD] = size(img);
    carved = zeros(dimY-1, dimX, dimD);
    
    for x=1:dimX
        % cut the seam out of the image
        carved(:,x,:) = img([1:horizSeam(x)-1,horizSeam(x)+1:dimY],x,:);
    end;