% Define the folders
input_folder = 'D:\STUDY\1PhD\Instant-NGP-for-RTX-3000-and-4000\Instant-NGP-for-RTX-3000-and-4000\data\nerf\mic_o\mic\test\';
output_folder = 'D:\STUDY\1PhD\Instant-NGP-for-RTX-3000-and-4000\Instant-NGP-for-RTX-3000-and-4000\data\nerf\mic\test\';
%input_folder = 'D:\STUDY\1PhD\Instant-NGP-for-RTX-3000-and-4000\Instant-NGP-for-RTX-3000-and-4000\data\nerf\hotdog_o\hotdog\test\';
%output_folder = 'D:\STUDY\1PhD\Instant-NGP-for-RTX-3000-and-4000\Instant-NGP-for-RTX-3000-and-4000\data\nerf\hotdog\test\';

% Create the output folder if it does not exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(input_folder, 'r_*.png');
theFiles = dir(filePattern);

for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
     % Skip files with '_depth' in their name
    if contains(baseFileName, '_depth')
        continue;
    end
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);

    % Read the image
    in_img = imread(fullFileName);
    mask = any(in_img, 3);
    alphaChannel = double(mask);
    
    
    % HSV 
    hsv_img = rgb2hsv(in_img);
    [height, width, channels] = size(in_img);


    %%%%%%%%%%%%%%%%%%%%%%%%%% noise cancel 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
grayImg =hsv_img(:,:,1);
threshold1 = 0.78;
threshold2 = 0.01;
localStdDev = stdfilt(grayImg, ones(5,5)); % ??3x3??
stdThreshold = 0.05; 
isHighNoise = grayImg > threshold1 & localStdDev > stdThreshold & mask;
isLowNoise = grayImg < threshold2 & mask;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
saturationChannel = hsv_img(:,:,2);
maskedSaturation = saturationChannel(mask);
thresh = graythresh(maskedSaturation);  
adjustedThresh = thresh * 0.4; 
binarySaturation = imbinarize(saturationChannel, adjustedThresh);
isHighNoise(mask) = isHighNoise(mask) | ~binarySaturation(mask);
initialHighNoise = isHighNoise;

needsFiltering = false(size(grayImg));

for row = 1:size(grayImg, 1)
    for col = 1:size(grayImg, 2)
        if isHighNoise(row, col) || isLowNoise(row, col)
            r1 = max(row - 4, 1);
            r2 = min(row + 4, size(grayImg, 1));
            c1 = max(col - 4, 1);
            c2 = min(col + 4, size(grayImg, 2));
            neighborhood = grayImg(r1:r2, c1:c2);
            neighborhoodMask = mask(r1:r2, c1:c2);

            validPixels = neighborhood(~(isHighNoise(r1:r2, c1:c2) | isLowNoise(r1:r2, c1:c2)) & neighborhoodMask & neighborhood > 0);
            % distance
            [rows, cols] = find(~(isHighNoise(r1:r2, c1:c2) | isLowNoise(r1:r2, c1:c2)) & neighborhoodMask);
            distances = sqrt((rows - (row-r1+1)).^2 + (cols - (col-c1+1)).^2);
            
            % select 2-4 values
            if length(validPixels) >= 5
                [~, idx] = sort(distances);
                selectedPixels = validPixels(idx(1:min(9, length(idx))));
                medianValue = median(selectedPixels);
                grayImg(row, col) = medianValue;
                %meanValue = mean(selectedPixels);
                %grayImg(row, col) = meanValue;
                isHighNoise(row, col) = false;
                isLowNoise(row, col) = false;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%% danger area%%%%%%%%%%%%%%%%%%%%%%%%%%%%

countReplacedPixels = 0;
replacedPixelCoords = zeros(20, 2);

CC = bwconncomp(initialHighNoise);
numPixels = cellfun(@numel,CC.PixelIdxList);


for i = 1:length(numPixels)
    if numPixels(i) > 20*20 
        idx = CC.PixelIdxList{i};
        [rows, cols] = ind2sub(size(grayImg), idx);
        
        expandedRows = max(min(rows)-3, 1):min(max(rows)+3, size(grayImg, 1));
        expandedCols = max(min(cols)-3, 1):min(max(cols)+3, size(grayImg, 2));
        
        %mask
        expandedMask = false(size(grayImg));
        expandedMask(expandedRows, expandedCols) = true;
        expandedMask(idx) = false;

        edgePixels = grayImg(expandedMask & mask);
        minEdge = min(edgePixels(:));
        maxEdge = max(edgePixels(:));
        meanEdge = mean(edgePixels(:));

        for idxPixel = idx'
            if mask(idxPixel) 
                pixelValue = grayImg(idxPixel);
                if pixelValue < minEdge || pixelValue > maxEdge
                    grayImg(idxPixel) = meanEdge;
                    countReplacedPixels = countReplacedPixels + 1;
                    if countReplacedPixels <= 20
                        [row, col] = ind2sub(size(grayImg), idxPixel);
                        replacedPixelCoords(countReplacedPixels, :) = [row, col];
                    end
                end
            end
        end
        outerMask = false(size(grayImg));
        outerRows = max(min(rows)-5, 1):min(max(rows)+5, size(grayImg, 1));
        outerCols = max(min(cols)-5, 1):min(max(cols)+5, size(grayImg, 2));
        outerMask(outerRows, outerCols) = true;
        outerMask(expandedMask) = false;  

        transitionPixels = grayImg(expandedMask & ~mask);
        outerPixels = grayImg(outerMask);

        [outerRows, outerCols] = find(outerMask);

        for idxPixel = find(expandedMask & ~mask)'
            pixelValue = grayImg(idxPixel);
            [row, col] = ind2sub(size(grayImg), idxPixel);

            distances = sqrt((outerRows - row).^2 + (outerCols - col).^2);
            [~, nearestIdx] = min(distances);

            nearestPixelValue = grayImg(outerRows(nearestIdx), outerCols(nearestIdx));

            weight = distances(nearestIdx);
            grayImg(idxPixel) = round((1-weight)*pixelValue + weight*nearestPixelValue);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    hsv_img(:,:,1) = grayImg;
    hsv_img(repmat(~mask, [1, 1, size(hsv_img, 3)])) = 0;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%hsv sat median%%%%%%%%%%%%%%%%%%
    saturationChannel=hsv_img(:,:,2) ;
    sat_validPixels = saturationChannel(mask);
    medians = zeros(1, 10);

    % 10 median
    for i = 1:10
        sample = randsample(sat_validPixels, min(1000, numel(sat_validPixels)), true);
        medians(i) = median(double(sample));
    end
    meanMedian = mean(medians);
    saturationChannel(mask) = meanMedian;
    hsv_img(:,:,2) = saturationChannel;


    img= hsv2rgb(hsv_img);


    %%%%%%%%%%%%%%%%%%%%%%%%%% CIELAB L & HSV Value error  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cielab_img = rgb2lab(in_img);
    % ? CIELAB L* 
    scaled_L = cielab_img(:,:,1) / 100;

    % error = CIELAB L* - HSV Value 
    error = abs(scaled_L - hsv_img(:,:,3)); 
    %error value resale
    hsv_value_max = max(hsv_img(repmat(mask, [1 1 3])));
    hsv_value_min = min(hsv_img(repmat(mask, [1 1 3])));
    error_min = min(error(mask));
    error_max = max(error(mask));
    error_rescaled = (error - error_min) / (error_max - error_min) * (hsv_value_max - hsv_value_min) + hsv_value_min;


    % error_rescaled mask
    error_rescaled(~mask) = error(~mask);


    %%%%%%%%%%%%%%%%%%%%%%% mean %%%%%%%%%%%%%%%%%%%%%%%%
    filterSize = [8 8]; 
    meanFilter = fspecial('average', filterSize);
    selectedRegion = error_rescaled(mask);
    filteredRegion = imfilter(selectedRegion, meanFilter, 'replicate');
    error_rescaled(mask) = filteredRegion;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%%%%%%%%%%%%%%%%%%%% hue mean filter %%%%%%%%%%%%%%%%%%%
%{
    hue = hsv_img(:,:,1);
    filterSize = 15;
    meanFilter = fspecial('average', filterSize);
    blurredHue = imfilter(hue, meanFilter, 'replicate');
    hue(mask) = blurredHue(mask);

    hsv_img(:,:,1) = hue;
%}
    bal_img = 0.65 * hsv_img(:,:,3) + 0.35*error_rescaled;


    %%%%%%%%%%%%%%%%%%%%%%%%%bal_img%%%%%%%%%%%%%%%%%%%%%%%%%%%
    masked_img = bal_img(mask);

    % ????????????
    max_val = max(masked_img(:));
    min_val = min(masked_img(:));
    median_val = median(masked_img(:));

    % ???????????????0.6
if (max_val - min_val) > 0.4
        % ???????????
    new_max = median_val + 0.2;
    new_min = median_val - 0.2;

        % ?????
    img_out = bal_img;
    img_out(mask) = (masked_img - min_val) / (max_val - min_val) * (new_max - new_min) + new_min;

        % ??????0?1??
    img_out(img_out > 1) = 1;
    img_out(img_out < 0) = 0;
else
    img_out = bal_img;
end
bal_img = img_out;
    
    
    hsv_img(:,:,3)=bal_img;
    % Convert back to RGB and save
    result = hsv2rgb(hsv_img);
    outputFileName = fullfile(output_folder, baseFileName);
    imwrite(result, outputFileName, 'Alpha', alphaChannel);
end