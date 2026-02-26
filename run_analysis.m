% =========================================================================
% Practical 1: 2D Convolution Analysis
% =========================================================================
%
% GROUP NUMBER: 18 
%
% MEMBERS:
%   - Kamil Singh, SNGKAM012
%   - Muhammed Zaakir Vahed , VHDMUH004


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
    output = edged_image;
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
    

%defining edge detection kernals 
    Gx = [-1  0  1;
          -2  0  2;
          -1  0  1];

    Gy = [1  2  1;
          0  0  0;
         -1 -2 -1];

%Load all images
    image_folder = 'sample_images';

    image_names = { ...
    'image_128x128.png', ...
    'image_256x256.png', ...
    'image_512x512.png', ...
    'image_1024x1024.png', ...
    'image_2048x2048.png'};

    num_images = 5;

% Preallocate result arrays 
    manual_times = zeros(num_images,1);
    builtin_times = zeros(num_images,1);
    speedups = zeros(num_images,1);

    fprintf('\n===== BENCHMARK RESULTS =====\n');

    for k = 1:5
    
    % Get image name
        filename = image_names{k};
        filepath = fullfile(image_folder, filename);
    
    % Read image
        img = imread(filepath);
        
    
    % Manual timing 
        for r = 1:5
            tic;
            manual_output = my_conv2(img, Gx, Gy);
            t_manual(r) = toc;
        end
        manual_times(k) = mean(t_manual);
    
    %Built-in timing
        for r = 1:5
            tic;
            builtin_output = inbuilt_conv2(img, Gx, Gy);
            t_builtin(r) = toc;
        end
        builtin_times(k) = mean(t_builtin);

    
    % Speedup
        speedups(k) = manual_times(k) / builtin_times(k);
    
    %Correctness check
        error_value = norm(double(manual_output) - double(builtin_output));
    
        fprintf('\nImage: %s\n', filename);
        fprintf('Manual Time:   %f s\n', manual_times(k));
        fprintf('Built-in Time: %f s\n', builtin_times(k));
        fprintf('Speedup:       %f\n', speedups(k));
        fprintf('Output Error:  %f\n', error_value);
    end
    %Plot execution times for each image 
figure;

bar([manual_times builtin_times]);

xlabel('Image Index');
ylabel('Execution Time (seconds)');
title('Manual vs Built-in Convolution Execution Time');

legend('Manual', 'Built-in');
grid on;

end
run_analysiss();