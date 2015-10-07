% Finds the horizontal seam by choosing the minimum accumulated energy
% value in the last column of the accumulated energy map, then backtracing
% to the first column to find the minimum horizontal seam
%
% input
% -----
% energyMap : 2-d array of the horizontal accumulated energy map
%
% output
% ------
% horizSeam : 1-d array with row indexes indicating where the horizontal
%             seams are
function horizSeam = findHorizSeam(energyMap)
    [dimY, dimX] = size(energyMap);
    horizSeam = zeros(1,dimX);
    [minV, minRow] = min(energyMap(:,dimX));
    y = minRow;
    horizSeam(dimX) = y;
    
    % start from the last column with the smallest value in the accumulated 
    % energy map, then for each pixel, choose the connected pixel that has 
    % the minimum value
    for x=dimX-1:-1:2
        if (y == 1)
            [minV, minPos] = min([energyMap(y,x-1), energyMap(y+1,x-1)]);
            if (minPos == 2)
                y = y + 1;
            end;
        elseif (y == dimY)
            [minV, minPos] = min([energyMap(y,x-1), energyMap(y-1,x-1)]);
            if (minPos == 2)
                y = y - 1;
            end;
        else
            [minV, minPos] = min([energyMap(y,x-1), energyMap(y-1,x-1), energyMap(y+1,x-1)]);
            if (minPos == 2)
                y = y - 1;
            elseif (minPos == 3)
                y = y + 1;
            end;
        end;
        horizSeam(x) = y;
    end;
    
    % rest is for the first column; choose the pixel with minimum energy value
    x = 1;
    y = horizSeam(2);
    
    if (y == 1)
        [minV, minPos] = min([energyMap(y,x), energyMap(y+1,x)]);
        if (minPos == 2)
            y = y + 1;
        end;
    elseif (y == dimY)
        [minV, minPos] = min([energyMap(y,x), energyMap(y-1,x)]);
        if (minPos == 2)
            y = y - 1;
        end;
    else
        [minV, minPos] = min([energyMap(y,x), energyMap(y-1,x), energyMap(y+1,x)]);
        if (minPos == 2)
            y = y - 1;
        elseif (minPos == 3)
            y = y + 1;
        end;
    end;
    
    horizSeam(1) = y;