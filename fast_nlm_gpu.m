function denoised_data = fast_nlm_gpu(data, h, search_window, patch_size)
    % GPU 加速的 Fast Non-Local Means (F-NLM) 去噪
    % 适用于 2D/3D 图像
    % 
    % 输入：
    %   - data: 2D/3D 图像 (double 或 uint8)
    %   - h: 平滑参数，控制去噪强度
    %   - search_window: 搜索窗口大小
    %   - patch_size: 计算相似性的块大小
    %
    % 输出：
    %   - denoised_data: 2D 或 3D 去噪图像 (uint8)
    if iscell(data)  % 2D 图像
        denoised_data = cell(size(data));
        for i = 1:numel(data)
            denoised_data{i} = fast_nlm_2d_gpu_integral(data{i}, h, floor(search_window/2), floor(patch_size/2));
            % denoised_data{i} = fast_nlm_2d_gpu(data{i}, h, floor(search_window/2), floor(patch_size/2));
            % denoised_data{i} = single(data{i});
        end
    else  % 3D 体数据
        denoised_data = fast_nlm_3d_gpu_integral(data, h, floor(search_window/2), floor(patch_size/2));
    end
end

function denoised_img = fast_nlm_2d_gpu(img, h, half_search, half_patch)
% nlm_gpu_vectorized - 利用 GPU 向量化加速的非局部均值（NLM）降噪算法
%
% 语法:
%   denoised_img = nlm_gpu_vectorized(img, h, half_search, half_patch)
%
% 输入:
%   img        - 输入图像（灰度图，double 类型）
%   h          - 衰减参数
%   half_search- 搜索窗口半径，候选位移范围为 [-half_search, half_search]
%   half_patch - patch 半径，patch 尺寸为 (2*half_patch+1)
%
% 输出:
%   denoised_img - 降噪后的图像
%
% 说明:
%   1. 对于候选位移 (dx,dy) ∈ [-half_search, half_search]×[-half_search, half_search]，
%      一次性计算原图与平移图之间的平方差（3D 数组，第三维代表不同位移）。
%   2. 使用大小为 (2*half_patch+1) 的卷积核对每个候选位移下的差异图进行卷积，
%      得到 patch 内的累积差异。
%   3. 利用权重 w = exp(-diff/(h^2)) 计算各候选位置的贡献，最后对所有候选位移加权平均。
%
% 注意：需要 Parallel Computing Toolbox 支持 GPU 计算。

    % 转换输入图像到 GPU，并确保为 double 类型
    img_gpu = gpuArray(single(img));
    mask = ones(size(img_gpu),'like', img_gpu);
    img_gpu = padarray(img_gpu, [half_search + half_patch, half_search + half_patch], 0, 'both');
    mask = padarray(mask, [half_search + half_patch, half_search + half_patch], 0, 'both');
    [M, N] = size(img_gpu);
    
    % 候选位移范围
    shifts = -half_search-half_patch:half_search+half_patch;
    [DX, DY] = meshgrid(shifts, shifts);
    numShifts = numel(DX);
    dx = DX(:);
    dy = DY(:);
    
    % 预分配存储所有平移后的图像 (3D 数组：M x N x numShifts)
    shifted_imgs = zeros(M, N, numShifts, 'like', img_gpu);
    shifted_mask = zeros(M, N, numShifts, 'like', mask);
    
    % 对每个候选位移计算平移后的图像
    for k = 1:numShifts
        shifted_imgs(:,:,k) = circshift(img_gpu, [dx(k), dy(k)]);
        shifted_mask(:,:,k) = circshift(mask, [dx(k), dy(k)]);
    end
    
    % 计算原图与各平移图之间的平方差
    % 这里利用广播机制省去repmat
    diff_sq = (img_gpu - shifted_imgs).^2;

    % 定义 patch 卷积核（box filter）
    patch_kernel = ones(2*half_patch+1, 2*half_patch+1, 'like', img_gpu) / (2*half_patch+1)^2;
    
    % 对 diff_sq 的每一“页”进行卷积，累加 patch 内的差异
    % imfilter 对 3D 数组会自动对每一页独立处理
    dist = convn(diff_sq, patch_kernel, 'same');
    
    % 计算权重： w = exp(-dist/(h^2))
    weights = exp( - dist / (h^2) );
    
    % 对各候选位移下的平移图像加权
    weighted_imgs = weights .* shifted_imgs .* shifted_mask;
    
    % 在第三维（候选位移维度）上求和
    weighted_sum = sum(weighted_imgs, 3);
    weight_sum   = sum(weights, 3);
    
    % 得到最终降噪图像
    denoised_img = weighted_sum ./ weight_sum;

    denoised_img = denoised_img(half_patch+half_search+1:end-half_patch-half_search, ...
        half_patch+half_search+1:end-half_patch-half_search);
    % 转回 CPU 数组（可选）
    denoised_img = gather(denoised_img);
end

function denoised_img = fast_nlm_2d_gpu_integral(img, h, half_search, half_patch)
    % 使用 GPU 加速的 NLM 降噪（积分图加速）

    img_gpu = gpuArray(single(img));
    mask = ones(size(img_gpu), 'like', img_gpu);
    img_gpu = padarray(img_gpu, [half_search + half_patch, half_search + half_patch], 0, 'both');
    mask = padarray(mask, [half_search + half_patch, half_search + half_patch], 0, 'both');
    [M, N] = size(img_gpu);
    
    shifts = -half_search-half_patch:half_search+half_patch;
    [DX, DY] = meshgrid(shifts, shifts);
    numShifts = numel(DX);
    dx = DX(:);
    dy = DY(:);
    
    weighted_img_sum = zeros(M, N, 'like', img_gpu);
    weight_sum = zeros(M, N, 'like', mask);
    
    for k = 1:numShifts
        % 平移图像
        shifted_img = circshift(img_gpu, [dx(k), dy(k)]);
        shifted_mask = circshift(mask, [dx(k), dy(k)]);
        
        % 计算平方误差并构建积分图
        diff_sq = (img_gpu - shifted_img).^2;
        int_diff_sq = cumsum(cumsum(diff_sq, 1), 2);
        
        % 计算 patch 距离（利用积分图 O(1) 查询）
        int_diff_sq = padarray(int_diff_sq, [1, 1], 0, 'pre');
        x1 = (1:(M-2*half_patch))+1;
        y1 = (1:(N-2*half_patch))+1;
        x2 = x1 + 2*half_patch;
        y2 = y1 + 2*half_patch;

        dist = (int_diff_sq(x2, y2) - int_diff_sq(x1-1,y2) - int_diff_sq(x2,y1-1) + int_diff_sq(x1-1,y1-1))/(2*half_patch+1)^2;
        dist = padarray(dist, [half_patch, half_patch],0,'both');

        % 计算权重
        weights = exp(-dist / (h^2));
        
        % 加权求和
        weighted_img_sum = weighted_img_sum + weights .* shifted_img .* shifted_mask;
        weight_sum = weight_sum + weights;
    end
    
    % 计算最终降噪图像
    denoised_img = weighted_img_sum ./ weight_sum;
    
    % 移除填充区域
    denoised_img = denoised_img(half_patch+half_search+1:end-half_patch-half_search, ...
                                half_patch+half_search+1:end-half_patch-half_search);
    
    denoised_img = gather(denoised_img); % 转回 CPU
end

function denoised_img = fast_nlm_3d_gpu_integral(img, h, half_search, half_patch)
    % 使用 GPU 加速的 NLM 降噪（积分图加速）

    img_gpu = gpuArray(single(img));
    mask = ones(size(img_gpu), 'like', img_gpu);
    img_gpu = padarray(img_gpu, [half_search + half_patch, half_search + half_patch, half_search + half_patch], 0, 'both');
    mask = padarray(mask, [half_search + half_patch, half_search + half_patch, half_search + half_patch], 0, 'both');
    [M, N, P] = size(img_gpu);
    
    shifts = -half_search-half_patch:half_search+half_patch;
    [DX, DY, DZ] = meshgrid(shifts, shifts, shifts);
    numShifts = numel(DX);
    dx = DX(:);
    dy = DY(:);
    dz = DZ(:);
    
    weighted_img_sum = zeros(M, N, P, 'like', img_gpu);
    weight_sum = zeros(M, N, P, 'like', mask);
    
    progress_bar = waitbar(0, 'Processing ...');

    for k = 1:numShifts
        str = sprintf('Processing... %d%%', round(k / numShifts * 100));
        waitbar(k/numShifts, progress_bar, str)

        % 平移图像
        shifted_img = circshift(img_gpu, [dx(k), dy(k), dz(k)]);
        shifted_mask = circshift(mask, [dx(k), dy(k), dz(k)]);
        
        % 计算平方误差并构建积分图
        diff_sq = (img_gpu - shifted_img).^2;
        int_diff_sq = cumsum(cumsum(cumsum(diff_sq, 1), 2), 3);
        
        % 计算 patch 距离（利用积分图 O(1) 查询）
        int_diff_sq = padarray(int_diff_sq, [1, 1, 1], 0, 'pre');
        x1 = (1:(M-2*half_patch))+1;
        y1 = (1:(N-2*half_patch))+1;
        z1 = (1:(P-2*half_patch))+1;
        x2 = x1 + 2*half_patch;
        y2 = y1 + 2*half_patch;
        z2 = z1 + 2*half_patch;

        dist = (int_diff_sq(x2, y2, z2) ...
            - int_diff_sq(x1-1, y2, z2) - int_diff_sq(x2, y1-1, z2) - int_diff_sq(x2, y2, z1-1) ...
            + int_diff_sq(x1-1, y1-1, z2) + int_diff_sq(x1-1, y2, z1-1) + int_diff_sq(x2, y1-1, z1-1) ...
            - int_diff_sq(x1-1, y1-1, z1-1) ) / (2*half_patch+1)^3;
        dist = padarray(dist, [half_patch, half_patch, half_patch], 0, 'both');

        % 计算权重
        weights = exp(-dist / (h^2));
        
        % 加权求和
        weighted_img_sum = weighted_img_sum + weights .* shifted_img .* shifted_mask;
        weight_sum = weight_sum + weights;
    end
    delete(progress_bar);
    
    % 计算最终降噪图像
    denoised_img = weighted_img_sum ./ weight_sum;
    
    % 移除填充区域
    denoised_img = denoised_img(half_patch+half_search+1:end-half_patch-half_search, ...
                                half_patch+half_search+1:end-half_patch-half_search, ...
                                half_patch+half_search+1:end-half_patch-half_search);
    
    denoised_img = gather(denoised_img); % 转回 CPU
end

