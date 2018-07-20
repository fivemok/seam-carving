% Cut the vertical seams from the source image
%
% input
% -----
% img : 3-d array of the source image
% horizSeam : 1-d array of the vertical seams
% 
% output
% ------
% carved : 3-d array of the image with the seams removed
function carved = vertSeamCarve(img, vertSeam)
    [dimY, dimX, dimD] = size(img);
    carved = zeros(dimY, dimX-1, dimD);
    
    for y=1:dimY
        % cut the seam out of the image
        carved(y,:,:) = img(y,[1:vertSeam(y)-1,vertSeam(y)+1:dimX],:);
    end;