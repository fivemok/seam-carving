% Cut n vertical seams out of the image
%
% input
% -----
% img : 3-d array of the source image
% n : number of seams to cut
%
% output
% ------
% carved : 3-d array of the source image with seams removed
function carved = carveVertSeams(img, n)
    carved = img;
    
    for seam=1:n
        vert = accMinVertPathMap(carved); % update accumulated energy map
        v = findVertSeam(vert); % find seam
        carved = vertSeamCarve(carved, v); % cut seam from img
    end;