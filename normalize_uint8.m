function normalized_img = normalize_uint8(img, lower_percentile, upper_percentile)
    % 归一化 2D 或 3D 数组到 0-255，使用百分比归一化
    %
    % 参数:
    %   img - 输入的 2D 或 3D 数组
    %   lower_percentile - 下百分比阈值（默认 0.03%）
    %   upper_percentile - 上百分比阈值（默认 99.7%）
    %
    % 返回:
    %   normalized_img - 归一化后的 uint8 图像

    if nargin < 2
        lower_percentile = 0.03;
    end
    if nargin < 3
        upper_percentile = 99.7;
    end

    % 将数据展平以计算百分位
    img_flat = img(:);
    
    % 计算分位数对应的最小值和最大值
    min_val = prctile(img_flat, lower_percentile);
    max_val = prctile(img_flat, upper_percentile);

    % 限制范围并归一化
    img_clipped = min(max(img, min_val), max_val); % 裁剪
    normalized_img = (img_clipped - min_val) / (max_val - min_val) * 255; % 归一化

    % 转换为 uint8
    normalized_img = uint8(normalized_img);
end
