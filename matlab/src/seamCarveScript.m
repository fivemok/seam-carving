img = imread('../images/flower.jpg');
img = double(img) / 255.0;
carved = carveVertSeams(img, 200);
%carved = carveVertSeams(carved, 100);
%carved = carveHorizSeams(img, 150);
%carved = carveHorizSeams(carved, 100);
imshow(carved);