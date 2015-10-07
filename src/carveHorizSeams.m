% Cut n horizontal seams out of the image
%
% input
% -----
% img : 3-d array of the source image
% n : number of seams to cut
%
% output
% ------
% carved : 3-d array of the source image with seams removed
function carved = carveHorizSeams(img, n)
    carved = img;
    
    for seam=1:n
        horiz = accMinHorizPathMap(carved); % update accumulated energy map
        h = findHorizSeam(horiz); % find seam
        [carved, horiz] = horizSeamCarve(carved, horiz, h); % cut seams from img
    end;