img = imread('../images/castle.jpg');
img = double(img) / 255.0;
%carved = carveVertSeams(img, 200);
carved = carveVertSeams(carved, 200);
%carved = carveHorizSeams(img, 100);
%carved = carveHorizSeams(carved, 100);
imshow(carved);