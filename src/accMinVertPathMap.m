% Compute the vertical accumulated energy map, starting at each pixel
% in the second row, then iterating to the last row, choosing the 
% minimum of its upper, upper left or upper right neighboring pixel's 
% accumuted energy value and summing it with its own gradient magnitude
%
% input
% -----
% img : 3-d array of the source image
% 
% output
% ------
% acc : 2-d array of the accumulated horzontal energy map
function acc = accMinVertPathMap(img)
    % compute magnitudes of gradients of img
    mag = rgb2gray(img);
    [dy, dx] = gradient(mag);
    mag = hypot(dy,dx);
    [dimY,dimX] = size(mag);
    acc = mag;
    
    % iterate starting from the second row, adding the value at the 
    % current pixel and the minimum of the pixels that are 'connected'
    % from the preceding row
    for y=2:dimY
        for x=1:dimX
            if (x == 1)
                acc(y,x) = mag(y,x) + min([acc(y-1,x), acc(y-1,x+1)]);
            elseif (x == dimX)
                acc(y,x) = mag(y,x) + min([acc(y-1,x), acc(y-1,x-1)]);
            else
                acc(y,x) = mag(y,x) + min([acc(y-1,x-1), acc(y-1,x), acc(y-1,x+1)]);
            end;
        end;
    end;
    