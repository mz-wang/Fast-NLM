function slices = extract_slice_from_3d(vol, slice_indices, slice_axis)
    % 从 3D 体数据中提取 2D 切片
    %
    % vol: 输入的 3D 体数据
    % slice_indices: 指定切片索引（单张或多张）
    % slice_axis: 切片方向 ('axial', 'sagittal', 'coronal')

    % 确保 slice_axis 合法
    if nargin < 3
        slice_axis = 'transverse';  % 默认轴向切片
    end

    % 获取数据尺寸
    vol_size = size(vol);
    
    % 选择切片方向
    switch slice_axis
        case 'transverse'     % XY 平面，切 Z 轴
            dim = 3;
        case 'coronal'  % YZ 平面，切 X 轴
            dim = 1;
        case 'sagittal'   % XZ 平面，切 Y 轴
            dim = 2;
        otherwise
            error('无效的切片方向，应为 "transverse", "sagittal" 或 "coronal"');
    end

    % 默认切片索引：取中间切片
    if nargin < 2 || isempty(slice_indices)
        slice_indices = round(vol_size(dim) / 2);
    end

    % % 获取 2D 切片
    % if isscalar(slice_indices)  % 单张切片
    %     slice = extract_single_slice(vol, slice_indices, dim);
    % else  % 多张切片
    %     slice = cell(1, length(slice_indices));
    %     for i = 1:length(slice_indices)
    %         slice{i} = extract_single_slice(vol, slice_indices(i), dim);
    %     end
    % end
    slices = cell(1, length(slice_indices));
    for i = 1:length(slice_indices)
        slices{i} = extract_single_slice(vol, slice_indices(i), dim);
    end
end

function slice = extract_single_slice(vol, index, dim)
    % 按指定方向提取单张切片
    switch dim
        case 1
            slice = squeeze(vol(index, :, :));
            slice = permute(slice, [2, 1]);
        case 2
            slice = squeeze(vol(:, index, :));
            slice = permute(slice, [2, 1]);
        case 3
            slice = squeeze(vol(:, :, index));
    end
end
