clc; clear; close all;

%% 1. 读取 3D TIFF 文件
[file, path] = uigetfile('*.tif', '选择3D TIFF文件');
if isequal(file, 0)
    error('未选择文件！');
end
tiff_path = fullfile(path, file);
info = imfinfo(tiff_path);
num_slices = numel(info);

% 读取 TIFF 文件
raw_data = zeros(info(1).Height, info(1).Width, num_slices, 'double');
for i = 1:num_slices
    raw_data(:, :, i) = double(imread(tiff_path, i));
end
[numx, numy, numz] = size(raw_data);
disp(['成功加载 3D TIFF 文件: ', file]);

%% 2. 添加 Rician 噪声
threshold = quantile(raw_data(:), 0.997);
sigma = 0.01 * threshold;  % 设定噪声标准差（可调）
noisy_data = add_rician_noise(raw_data, sigma);

%% 3. 选择 2D 或 3D 数据
is_3d = false;
choice = input('处理 [1] 2D 还是 [2] 3D 数据？', 's');

if strcmp(choice, '1')
    % 处理 2D 数据
    slice_direction = input('请输入要提取的切片方向（[1] 横断面 [2] 冠状面 [3] 矢状面）： ');
    switch slice_direction
        case 1
            direction = 'transverse';
            num_slices = numz;
        case 2
            direction = 'coronal';
            num_slices = numx;
        case 3
            direction = 'sagittal';
            num_slices = numy;
        otherwise
            error('切片方向无效！');
    end
    slice_indices = input(['请输入要提取的切片索引（1 到 ', num2str(num_slices), '）：']);
    % 验证输入索引是否有效
    if any(slice_indices < 1 | slice_indices > num_slices)
        error('输入的切片索引超出范围！');
    end
    % 提取 2D 切片
    selected_data = extract_slice_from_3d(noisy_data, [slice_indices], direction);  % 使用默认轴向切片
    clean_data = extract_slice_from_3d(raw_data, [slice_indices], direction);

    for i = 1 : numel(slice_indices)
        selected_data{i} = normalize_uint8(selected_data{i});
        clean_data{i} = normalize_uint8(clean_data{i});
    end

    disp(['选择了 2D 切片: 第 ', num2str(slice_indices), ' 层']);
elseif strcmp(choice, '2')
    selected_data = normalize_uint8(noisy_data);
    clean_data = normalize_uint8(raw_data);
    is_3d = true;
    disp('选择了完整 3D 数据');
else
    error('无效输入！');
end

% 可视化噪声数据
figure;
if is_3d
    imshow(selected_data(:, :, round(num_slices/2)), []);
    title(['带 Rician 噪声的 3D 数据（中间切片）, transverse, index = ', num2str(round(num_slices/2))]);
else
    % title('带 Rician 噪声的 2D 数据');
    n_imgs = numel(selected_data);
    if n_imgs ==1
        imshow(selected_data{1}, []);
        title(['带 Rician 噪声的 2D 数据: ', direction,', index = ', num2str(slice_indices(1))]);
    else
        for i = 1:n_imgs
            subplot(1,n_imgs,i)
            imshow(selected_data{i}, []);
            title(['index = ', num2str(slice_indices(i))]);
        end
        sgtitle(['带 Rician 噪声的 2D 数据: ', direction]);
    end
end

%% ️4. 运行 快/慢 版本去噪
use_gpu = input('使用 [1] fast_NLM 还是 [2] NLM ?', 's');

h = 13;  % 去噪强度
search_window = 21;  % 搜索窗口大小
patch_size = 7;  % 相似度窗口大小

tic;  % 计时
if strcmp(use_gpu, '1')
    disp('正在使用 fast_NLM 进行去噪...');
    denoised_data = fast_nlm_gpu(selected_data, h, search_window, patch_size);
elseif strcmp(use_gpu, '2')
    disp('正在使用 NLM 进行去噪...');
    denoised_data = nlm(selected_data, h, search_window, patch_size);
else
    error('无效输入！');
end

if iscell(denoised_data)
    for i = 1 : numel(denoised_data)
        denoised_data{i} = normalize_uint8(denoised_data{i});
        % denoised_data{i} = uint8(denoised_data{i});
    end
else
    denoised_data = normalize_uint8(denoised_data);
end
denoising_time = toc;
disp(['去噪完成！耗时: ', num2str(denoising_time), ' 秒']);

%% 5. 计算 PSNR 和 SSIM
if is_3d
    psnr_value = psnr(denoised_data(:, :, round(num_slices/2)), clean_data(:, :, round(num_slices/2)));
    ssim_value = ssim(denoised_data(:, :, round(num_slices/2)), clean_data(:, :, round(num_slices/2)));
else
    psnr_value=zeros(1,n_imgs);
    ssim_value=zeros(1,n_imgs);
    for i=1:n_imgs
        psnr_value(i) = psnr(denoised_data{i}, clean_data{i});
        ssim_value(i) = ssim(denoised_data{i}, clean_data{i});
    end
end

disp(['PSNR: ', num2str(psnr_value)]);
disp(['SSIM: ', num2str(ssim_value)]);

% 显示去噪结果
figure;
if is_3d
    imshow(denoised_data(:, :, round(num_slices/2)), []);
    title(['去噪后的 3D 数据（中间切片）, transverse, index = ', num2str(round(num_slices/2))]);
else
    
    if n_imgs ==1
        imshow(denoised_data{1}, []);
        title(['去噪后的 2D 数据: ', direction,', index = ', num2str(slice_indices(1))]);
    else
        for i = 1:n_imgs
            subplot(1,n_imgs,i)
            imshow(denoised_data{i}, []);
            title(['index = ', num2str(slice_indices(i))]);
        end
        sgtitle(['去噪后的 2D 数据: ', direction]);
    end
end

disp('处理完成！');

