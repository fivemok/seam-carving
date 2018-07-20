% Compute the horizontal accumulated energy map, starting at each pixel
% in the second column, then iterating to the last column, choosing the 
% minimum of its left, upper left or lower left neighboring pixel's 
% accumuted energy value and summing it with its own gradient magnitude
%
% input
% -----
% img : 3-d array of the source image
% 
% output
% ------
% acc : 2-d array of the accumulated horzontal energy map
function acc = accMinHorizPathMap(img)
    % compute magnitudes of gradients of img
    mag = rgb2gray(img);
    [dy, dx] = gradient(mag);
    mag = hypot(dy,dx);
    [dimY,dimX] = size(mag);
    acc = mag;
    
    % iterate starting from the second column, adding the value at the 
    % current pixel and the minimum of the pixels that are 'connected'
    % from the preceding column
    for x=2:dimX
        for y=1:dimY
            if (y == 1)
                acc(y,x) = mag(y,x) + min([acc(y,x-1), acc(y+1,x-1)]);
            elseif (y == dimY)
                acc(y,x) = mag(y,x) + min([acc(y,x-1), acc(y-1,x-1)]);
            else
                acc(y,x) = mag(y,x) + min([acc(y-1,x-1), acc(y,x-1), acc(y+1,x-1)]);
            end;
        end;
    end;
    