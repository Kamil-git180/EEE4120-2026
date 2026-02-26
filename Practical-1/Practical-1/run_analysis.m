% =========================================================================
% Practical 1: 2D Convolution Analysis
% =========================================================================
%
% GROUP NUMBER: 18 
%
% MEMBERS:
%   - Kamil Singh, SNGKAM012
%   - Muhammed Zaakir Vahed , Student Number


%% ========================================================================
%  PART 1: Manual 2D Convolution Implementation
%  ========================================================================
%
% REQUIREMENT: You may NOT use built-in convolution functions (conv2, imfilter, etc.)

% TODO: Implement manual 2D convolution using Sobel Operator(Gx and Gy)
% output - Convolved image result (grayscale)


function edged_image = my_conv2(img, Gx, Gy)


%edge magnitude using Sobel operator manually
%   img : grayscale image matrix
%   Gx  : horizontal Sobel kernel
%   Gy  : vertical Sobel kernel
%
%   


    img = rgb2gray(img);


    % Get sizes
    [x, y] = size(img);          % Image size
    [gx, gy] = size(Gx);         % Kernel size (assume Gx and Gy are same size)
    
    % Calculate padding sizes since we want a similar size to the input 
    % Use the eqn n + 2p -f = n , where n- input image size, p - padding
    % needed , f size of filter  
    
    pad_x = floor(gx/2);
    pad_y = floor(gy/2);

  


    
    
    % Pading the input image using padarray function(please dont give me 0)
   new_img = padarray(img, [pad_x pad_y], 0, 'both');
    new_img = double(new_img);
    
    % Initialize output images for Gx and Gy
    edge_x = zeros(x, y);
    edge_y = zeros(x, y);

    Gx_flipped = Gx(end:-1:1, end:-1:1);
    Gy_flipped = Gy(end:-1:1, end:-1:1);
    
    % Perform convolution for both kernels using nested loop that goes
    % through each row and coloum of the padded image 
    for i = 1:x    %x and y being the lenghts of the image
        for j = 1:y
            patch_of_img = new_img(i:i+gx-1, j:j+gy-1);
            edge_x(i,j) = sum(sum(patch_of_img .* Gx_flipped));
            edge_y(i,j) = sum(sum(patch_of_img .* Gy_flipped));
        end
    end
    
    %  magnitude
    edged_image = sqrt(edge_x.^2 + edge_y.^2);
    

%threshold = 25;  % cuts off the initial spike
%edged_image = double(edged_image > threshold);

threshold = 2 * mean(edged_image(:));
edged_image = double(edged_image > threshold);


% Normalize first
% edged_image_norm = edged_image / max(edged_image(:));
% 
% % Mean of non-zero pixels only, then double it
% non_zero = edged_image_norm(edged_image_norm > 0);
% threshold = mean(non_zero) * 2;
% 
% edged_image = double(edged_image_norm > threshold);

    % Display the edge magnitude
    figure;
    imshow(edged_image);
    title('Grey-scale edge detection using sobel operator manually');
    


end



%% ========================================================================
%  PART 2: Built-in 2D Convolution Implementation
%  ========================================================================
%   
% REQUIREMENT: You MUST use the built-in conv2 function

% TODO: Use conv2 to perform 2D convolution
% output - Convolved image result (grayscale)
function output = inbuilt_conv2(img, Gx, Gy) %Add necessary input arguments
%   img : grayscale image matrix
%   Gx  : horizontal Sobel kernel
%   Gy  : vertical Sobel kernel

 img = rgb2gray(img);

 edge_x = conv2(img, Gx, 'same');  % horizontal gradient
 edge_y = conv2(img, Gy, 'same');  % vertical gradient


edged_image = sqrt(edge_x.^2 + edge_y.^2);
    % Display the magnitude

    threshold = 25;  % cuts off the initial spike
edged_image = double(edged_image > threshold);
    figure;
    imshow(edged_image, []);
    title('Grey-scale edge detection using in-built convolution matlab fn');
end


%% ========================================================================
%  PART 3: Testing and Analysis
%  ========================================================================
%
% Compare the performance of manual 2D convolution (my_conv2) with MATLAB's
% built-in conv2 function (inbuilt_conv2).

function run_analysiss()
    % TODO1:
    % Load all the sample images from the 'sample_images' folder
    
    % TODO2:
    % Define edge detection kernels (Sobel kernel)
    
    % TODO3:
    % For each image, perform the following:
    %   a. Measure execution time of my_conv2
    %   b. Measure execution time of inbuilt_conv2
    %   c. Compute speedup ratio
    %   d. Verify output correctness (compare results)
    %   e. Store results (image name, time_manual, time_builtin, speedup)
    %   f. Plot and compare results
    %   g. Visualise the edge detection results(Optional)
    
%hi Z just using this to test some stuff feel free to removei called the
%fns in here but idk if thats really required, not the F variable here uses
%a matlab inbuilt edge detector just used to compare results should
%probably remove

%defining edge detection kernals 
gx = [-1  0  1;
      -2  0  2;
      -1  0  1];

gy = [1 2 1;
      0  0  0;
      -1  -2  -1];

    I = imread ("sample_images\image_256x256" + ...
        ".png");
    T = rgb2gray(I);
    F = edge(T, 'sobel');
    figure; imshow(F);
   
    my_conv2(I, gx, gy );       %these are comming out quite similar pretty worrying 
   
    inbuilt_conv2(I, gx, gy)
    
    
end
run_analysiss();