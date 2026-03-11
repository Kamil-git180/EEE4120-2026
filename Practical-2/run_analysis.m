% =========================================================================
% Practical 2: Mandelbrot-Set Serial vs Parallel Analysis
% =========================================================================
%
% GROUP NUMBER:
%
% MEMBERS:
%   - Member 1 Name, Student Number
%   - Member 2 Name, Student Number

%% ========================================================================
%  PART 1: Mandelbrot Set Image Plotting and Saving
%  ========================================================================
%
% TODO: Implement Mandelbrot set plotting and saving function


%Should take input data from both computations and output function
%input data being resolution , iteration counts (image_data)
%the idea is once the image data is calcuulated in the computation
%function(s) it can just be plugged into plot to well plot!



%i added run anaylysis here for debuggign reasons, the map fn is below it
% using this to test you can ignore/remove

function run_analysis()

    image_sizes = [
        800,  600;
        1280,  720;
        1920, 1080;
        2048, 1080;
        2560, 1440;
        3840, 2160;
        5120, 2880;
        7680, 4320
    ];
    size_labels = {'SVGA','HD','Full HD','2K','QHD','4K','5K','8K'};

    % Ask user for worker count
    max_cores = feature('numcores');
    fprintf('Your machine has %d physical cores.\n', max_cores);
    num_workers = input(sprintf('How many workers do you want to use? (2-%d): ', max_cores));

    % Validate input
    if num_workers < 2 || num_workers > max_cores
        fprintf('Invalid input, defaulting to 2 workers.\n');
        num_workers = 2;
    end

    n_sizes = size(image_sizes, 1);
    time_serial   = zeros(n_sizes, 1);
    time_parallel = zeros(n_sizes, 1);

    for s = 1:n_sizes
        W = image_sizes(s, 1);
        H = image_sizes(s, 2);

        % Serial
        fprintf('Running serial: %s (%dx%d)...\n', size_labels{s}, W, H);
        t = tic;
        img_serial = mandelbrot_serial(H, W);
        time_serial(s) = toc(t);
        fprintf('  Serial time: %.2f s\n', time_serial(s));
        mandelbrot_plot(img_serial, W, H, 'serial');

        % Parallel
        fprintf('Running parallel: %s (%dx%d) with %d workers...\n', size_labels{s}, W, H, num_workers);
        t = tic;
        img_parallel = mandelbrot_parallel(H, W, num_workers);
        time_parallel(s) = toc(t);
        fprintf('  Parallel time: %.2f s\n', time_parallel(s));
        mandelbrot_plot(img_parallel, W, H, 'parallel');

        % Verify results match
        if isequal(img_serial, img_parallel)
            fprintf('  Verification: PASSED\n');
        else
            fprintf('  Verification: FAILED\n');
        end

        % Speedup
        speedup = time_serial(s) / time_parallel(s);
        efficiency = (speedup / num_workers) * 100;
        fprintf('  Speedup: %.2fx | Efficiency: %.1f%%\n', speedup, efficiency);
    end

    % Summary table
    fprintf('\n%-10s %-12s %-12s %-10s %-12s\n', 'Size', 'Serial(s)', 'Parallel(s)', 'Speedup', 'Efficiency');
    fprintf('%s\n', repmat('-', 1, 58));
    for s = 1:n_sizes
        speedup    = time_serial(s) / time_parallel(s);
        efficiency = (speedup / num_workers) * 100;
        fprintf('%-10s %-12.2f %-12.2f %-10.2f %-12.1f%%\n', ...
            size_labels{s}, time_serial(s), time_parallel(s), speedup, efficiency);
    end
end

%%%%%%%%

%intially tried to do with imagesc however no image would load, after some
%reasearch 'imwrite; was used 
%imwrite takes the pixel value from the normalized(0-256) image data and
%assigns it a colour like a heat map with higher number being the warmest
%mandelplot
function mandelbrot_plot(img_data, width, height, method_type)
    norm_data = uint8(255 * (img_data / max(img_data(:)))); %scales values down and spreads them between 0 -256 
    filename = sprintf('mandelbrot_%s_%dx%d.png', method_type, width, height);
    imwrite(norm_data, hot(256), filename);
    fprintf('Saved: %s\n', filename);
end



%% ========================================================================
%  PART 2: Serial Mandelbrot Set Computation
%  ========================================================================`
%
%Serial Mandelbrot set computation function
%inputs : width, height of image
%output : img data for mandelbrot_plot
function img_data = mandelbrot_serial(height, width) 

img_data = zeros(height, width); %empty matrix of the image
max_iter = 1000; %max iterations is 1000 for all image sizes

%Dividing region to fit into image resolution
 x_coord = linspace(-2.0, 0.5, width);   
 y_coord = linspace(-1.2, 1.2 , height);  


 %from the pseudocode
 for row = 1:height
     for col = 1:width
         x0 = x_coord(col);
         y0 = y_coord(row);

         x= 0 ;
         y = 0; 
         idx = 0;

         while (idx < max_iter) && (x^2 + y^2 <= 4)
             x_next = x^2 - y^2 + x0;
             y_next = 2*x*y + y0;

             idx = idx + 1;
             x = x_next;
             y = y_next;
         end
         img_data(row, col) = idx; % Store the iteration count for the pixel
     end
 end



end

%% ========================================================================
%  PART 3: Parallel Mandelbrot Set Computation
%  ========================================================================
%
%TODO: Parallel Mandelbrot set computation function
%inputs : width, height of image
          %number of works required (out of 8) 
%output : img data for mandelbrot_plot
function img_data = mandelbrot_parallel(height, width, num_workers) 
    
img_data = zeros(height, width); %empty matrix of the image
max_iter = 1000; %max iterations is 1000 for all image sizes

%Dividing region to fit into image resolution
 x_coord = linspace(-2.0, 0.5, width);   
 y_coord = linspace(-1.2, 1.2 , height); 


 pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= num_workers  %checks for active workers and if correct number of them needed
        if ~isempty(pool)
            delete(pool);       % shut down wrong-sized pool
        end
        parpool('local', num_workers); 
    end

%using parfor for the outer loop as it can be more efficient
    parfor row = 1:height
        row_data = zeros(1, width);   % local temp for this row so that each worker has its own independant row_data
        y0 = y_coord(row);            % sliced read

        for col = 1:width
            x0 = x_coord(col);        % broadcast read

            x = 0.0;
            y = 0.0;
            idx = 0;

            while (idx < max_iter) && (x^2 + y^2 <= 4)
                x_next = x^2 - y^2 + x0;
                y_next = 2*x*y      + y0;
                idx = idx + 1;
                x = x_next;
                y = y_next;
            end

            row_data(col) = idx; %each worker writes to its own row_data which all get merged together at the end
        end

        img_data(row, :) = row_data;  % write whole row at once
    end

end

%% ========================================================================
%  PART 4: Testing and Analysis
%  ========================================================================
% Compare the performance of serial Mandelbrot set computation
% with parallel Mandelbrot set computation.




%% using this to test you can ignore/remove
% 
% function run_analysis_mandel()
%   image_sizes = [
%     800, 600
% ];
% size_labels = {'SVGA'};
% 
%     n_sizes = size(image_sizes, 1);
%     time_serial = zeros(n_sizes, 1);
% 
%     for s = 1:n_sizes
%         W = image_sizes(s, 1);
%         H = image_sizes(s, 2);
%         fprintf('Running serial: %s (%dx%d)...\n', size_labels{s}, W, H);
% 
%         t = tic;
%         img_data = mandelbrot_serial(H, W);
%         time_serial(s) = toc(t);
% 
%         fprintf('  Time: %.2f s\n', time_serial(s));
% 
%    fprintf("Calling plot...\n");
% 
%         mandelbrot_plot(img_data, W, H, 'serial');
%     end
% 
%     fprintf('\n%-10s %10s\n', 'Size', 'Time(s)');
%     fprintf('%s\n', repmat('-', 1, 22));
%     for s = 1:n_sizes
%         fprintf('%-10s %10.2f\n', size_labels{s}, time_serial(s));
%     end
% end

    %TODO: For each image size, perform the following:
    %   a. Measure execution time of mandelbrot_serial
    %   b. Measure execution time of mandelbrot_parallel
    %   c. Store results (image size, time_serial, time_parallel, speedup)  
    %   d. Plot and save the Mandelbrot set images generated by both methods
    






