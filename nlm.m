function denoised_data = nlm(data, h, search_window, patch_size)
    % Fast Non-Local Means (F-NLM) 去噪
    % 适用于 2D/3D 图像数据
    % 
    % 输入：
    %   - data: 2D/3D 图像 (double 或 uint8 格式)
    %   - h: 平滑参数，控制去噪强度
    %   - search_window: 搜索窗口大小
    %   - patch_size: 计算相似性的块大小
    %
    % 输出：
    %   - denoised_data: 2D 或 3D 去噪图像
    
    if iscell(data)  % 2D 图像
        denoised_data = cell(size(data));
        for i = 1:numel(data)
            denoised_data{i} = nlm_2d(single(data{i}), h, search_window, patch_size);
        end
    else  % 3D 体数据
        denoised_data = nlm_3d(single(data), h, search_window, patch_size);
    end
end

function denoised_img = nlm_2d(img, h, search_window, patch_size)
    % 处理 2D 图像的非局部均值去噪
    [rows, cols] = size(img);
    denoised_img = zeros(rows, cols);

    half_patch = floor(patch_size / 2);
    half_search = floor(search_window / 2);
    
    [X, Y] = meshgrid(-half_patch:half_patch, -half_patch:half_patch);
    gauss_weights = exp(-(X.^2 + Y.^2) / (2 * half_patch^2));

    for i = 1:rows
        for j = 1:cols
            r_min = max(i - half_search, 1);
            r_max = min(i + half_search, rows);
            c_min = max(j - half_search, 1);
            c_max = min(j + half_search, cols);
            
            ref_patch = get_patch(img, i, j, half_patch);
            weights = zeros(r_max - r_min + 1, c_max - c_min + 1);
            total_weight = 0;
            weighted_sum = 0;

            for r = r_min:r_max
                for c = c_min:c_max
                    if r == i && c == j
                        continue;
                    end
                    
                    cmp_patch = get_patch(img, r, c, half_patch);
                    dist = sum(sum(gauss_weights .* (ref_patch - cmp_patch).^2));
                    w = exp(-dist / (h^2));
                    
                    weights(r - r_min + 1, c - c_min + 1) = w;
                    total_weight = total_weight + w;
                    weighted_sum = weighted_sum + w * img(r, c);
                end
            end
            
            if total_weight > 0
                denoised_img(i, j) = weighted_sum / total_weight;
            else
                denoised_img(i, j) = img(i, j);
            end
        end
    end
end

function denoised_vol = nlm_3d(vol, h, search_window, patch_size)
    % 处理 3D 体数据的非局部均值去噪
    [rows, cols, slices] = size(vol);
    denoised_vol = zeros(rows, cols, slices);

    half_patch = floor(patch_size / 2);
    half_search = floor(search_window / 2);

    [X, Y, Z] = meshgrid(-half_patch:half_patch, -half_patch:half_patch, -half_patch:half_patch);
    gauss_weights = exp(-(X.^2 + Y.^2 + Z.^2) / (2 * half_patch^2));

    for i = 1:rows
        for j = 1:cols
            for k = 1:slices
                r_min = max(i - half_search, 1);
                r_max = min(i + half_search, rows);
                c_min = max(j - half_search, 1);
                c_max = min(j + half_search, cols);
                s_min = max(k - half_search, 1);
                s_max = min(k + half_search, slices);
                
                ref_patch = get_patch_3d(vol, i, j, k, half_patch);
                total_weight = 0;
                weighted_sum = 0;

                for r = r_min:r_max
                    for c = c_min:c_max
                        for s = s_min:s_max
                            if r == i && c == j && s == k
                                continue;
                            end
                            
                            cmp_patch = get_patch_3d(vol, r, c, s, half_patch);
                            dist = sum(sum(sum(gauss_weights .* (ref_patch - cmp_patch).^2)));
                            w = exp(-dist / (h^2));

                            total_weight = total_weight + w;
                            weighted_sum = weighted_sum + w * vol(r, c, s);
                        end
                    end
                end

                if total_weight > 0
                    denoised_vol(i, j, k) = weighted_sum / total_weight;
                else
                    denoised_vol(i, j, k) = vol(i, j, k);
                end
            end
        end
    end
end

function patch = get_patch(img, i, j, half_size)
    % 获取 2D 图像的块
    [rows, cols] = size(img);
    r_min = max(i - half_size, 1);
    r_max = min(i + half_size, rows);
    c_min = max(j - half_size, 1);
    c_max = min(j + half_size, cols);
    patch = img(r_min:r_max, c_min:c_max);
    
    pad_r = 2 * half_size + 1 - size(patch, 1);
    pad_c = 2 * half_size + 1 - size(patch, 2);
    patch = padarray(patch, [pad_r, pad_c], 'replicate', 'post');
end

function patch = get_patch_3d(vol, i, j, k, half_size)
    % 获取 3D 体数据的块
    [rows, cols, slices] = size(vol);
    r_min = max(i - half_size, 1);
    r_max = min(i + half_size, rows);
    c_min = max(j - half_size, 1);
    c_max = min(j + half_size, cols);
    s_min = max(k - half_size, 1);
    s_max = min(k + half_size, slices);
    
    patch = vol(r_min:r_max, c_min:c_max, s_min:s_max);
    
    pad_r = 2 * half_size + 1 - size(patch, 1);
    pad_c = 2 * half_size + 1 - size(patch, 2);
    pad_s = 2 * half_size + 1 - size(patch, 3);
    patch = padarray(patch, [pad_r, pad_c, pad_s], 'replicate', 'post');
end
