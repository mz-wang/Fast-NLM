function noisy_data = add_rician_noise(data, sigma)
    % 对2D / 3D图像数据添加 Rician 噪声
    % data: 
    %   - 3D数组 (直接传入 3D 体数据)
    %   - Cell数组 (包含多张 2D 图片)
    % sigma: 噪声标准差，控制噪声大小
    
    if iscell(data)  % 处理 2D cell 数据
        noisy_data = cell(size(data));
        for i = 1:numel(data)
            noisy_data{i} = add_rician_noise_single(data{i}, sigma);
        end
    else  % 处理 3D 体数据
        noisy_data = add_rician_noise_single(data, sigma);
    end
end

function noisy_img = add_rician_noise_single(img, sigma)
    % 向单个2D/3D图像添加 Rician 噪声
    N1 = sigma * randn(size(img));  % 生成 N1 高斯噪声
    N2 = sigma * randn(size(img));  % 生成 N2 高斯噪声
    noisy_img = sqrt((img + N1).^2 + N2.^2);  % 计算 Rician 噪声
end
