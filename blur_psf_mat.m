clear;close all;
%% settings
folder_image = 'H:\experiment\test_new';
folder_kernel = 'H:\experiment\kernel';
blur_mat = 'H:\experiment\new_blurmat';
blur_pic = 'H:\experiment\new_blurpic';
%% generate data
filepaths = dir(fullfile(folder_image,'*.png'));
kernelpaths=dir(fullfile(folder_kernel,'*.mat'));
 num=0;   
for i = 1 : length(filepaths)
    for j=1:length(kernelpaths)
    
        image = imread(fullfile(folder_image,filepaths(i).name));

        load(fullfile(folder_kernel,kernelpaths(j).name));
        k=f;

        image = rgb2ycbcr(image);
        image_y= im2double(image(:, :, 1));
        image_cb=im2double(image(:, :, 2));
        image_cr=im2double(image(:, :, 3));
        x=image_y;
        NoiseLevel = 0.001; 
        image_blur_y=imfilter(image_y,k,'circular','conv','same')+NoiseLevel*randn(size(image_y));
        y=image_blur_y;
        clear image_blur
        image_blur(:,:,1)=y;
        image_blur(:,:,2)=image_cb;
        image_blur(:,:,3)=image_cr;
        image_blur=ycbcr2rgb(image_blur);
        num=num+1;

        imwrite(image_blur,sprintf('%s/im%02d_ker%02d.png',blur_pic,i,j));
        save(sprintf('%s/im%02d_ker%02d.mat', blur_mat, i,j),...
        'k', 'x', 'y','image_cb','image_cr');
        
         
       
   % imwrite(image_blur, blur_fu);
    %figure;imshow(image_blur);
    end
end
