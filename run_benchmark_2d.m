clc,clear,close all

img_name_list = ["pd_ai_msles2_1mm_pn0_rf20.tif", "pd_icbm_normal_1mm_pn0_rf0.tif", "pd_icbm_normal_1mm_pn0_rf20.tif", ...
    "pd_icbm_normal_1mm_pn0_rf40.tif", "t1_ai_msles2_1mm_pn0_rf20.tif", "t1_icbm_normal_1mm_pn0_rf0.tif", ...
    "t1_icbm_normal_1mm_pn0_rf20.tif", "t1_icbm_normal_1mm_pn0_rf40.tif", "t2_ai_msles2_1mm_pn0_rf20.tif", ...
    "t2_icbm_normal_1mm_pn0_rf0.tif", "t2_icbm_normal_1mm_pn0_rf20.tif", "t2_icbm_normal_1mm_pn0_rf40.tif"];
sigma_ratio_list = [0.002,0.005,0.01,0.02];
search_size_list=[15,21,27,31];
patch_size_list=[5,7,9,11];
h_list=[5,10,15,20,25];
direction_list=["transverse","coronal","sagittal"];
slice_idx_list=[-1];

total_progress = numel(img_name_list) * numel(sigma_ratio_list) * numel(search_size_list) * ...
                     numel(patch_size_list) * numel(h_list) * numel(direction_list) * numel(slice_idx_list);

img_folder = "D:\BaiduSyncdisk\课程\课程设计\data\tiff";
img_path_list = fullfile(img_folder, img_name_list);

clean_folder = img_folder + "\benchmark\clean";
noisy_folder = img_folder + "\benchmark\noisy";
denoised_folder = img_folder + "\benchmark\denoised";
csv_path = img_folder + "\benchmark\benchmark_results.csv";

% 检查文件夹是否存在
if ~exist(clean_folder, 'dir')
    mkdir(clean_folder);
end
if ~exist(noisy_folder, 'dir')
    mkdir(noisy_folder);
end
if ~exist(denoised_folder, 'dir')
    mkdir(denoised_folder);
end

% 定义表头（但无数据）
emptyTable = table([], [], [], [], [], [], [], [], [], [], [], [], [], ...
    'VariableNames', {'Modal', 'Type', 'Thickness', 'PN', 'RF', 'Direction', ...
    'Slice_Index', 'SR', 'Search', 'Patch', 'H', 'PSNR', 'SSIM'});

% 写入 CSV（清空文件内容但保留表头）
writetable(emptyTable, csv_path, 'Encoding', 'UTF-8');
progress_bar = waitbar(0, 'Processing ...');
k=0;
for i = 1:numel(img_path_list)

    img_path = img_path_list(i);

    info = imfinfo(img_path);
    num_slices = numel(info);

    % 读取 TIFF 文件
    raw_data = zeros(info(1).Height, info(1).Width, num_slices, 'double');
    for j = 1:num_slices
        raw_data(:, :, j) = double(imread(img_path, j));
    end
    [numx, numy, numz] = size(raw_data);
    disp(img_path);

    metadata = extract_metadata(img_name_list(i));
    for jdirection = 1:numel(direction_list)
        metadata("direction")=[direction_list(jdirection)];
        for jslice = 1:numel(slice_idx_list)
            metadata("slice")=[num2str(slice_idx_list(jslice))];
            direction = direction_list(jdirection);
            %is_3d = false;
            switch direction
                case 'transverse'
                    num_slices = numz;
                case 'coronal'
                    num_slices = numx;
                case 'sagittal'
                    num_slices = numy;
                otherwise
                    error('切片方向无效！');
            end
            slice_idx = slice_idx_list(jslice);
            if slice_idx == -1
                slice_idx = round(num_slices/2);
            end
            if slice_idx < 1 || slice_idx > num_slices
                error('输入的切片索引超出范围！');
            end
            clean_data = extract_slice_from_3d(raw_data, [slice_idx], direction);
            clean_data{1} = normalize_uint8(clean_data{1});

            clean_join =  [metadata("modal"),metadata("type"),metadata("thickness"),metadata("pn"),metadata("rf"), ...
                metadata("direction"),num2str(slice_idx)];
            clean_path = fullfile(clean_folder, strjoin(clean_join,"_")+".tif");
            imwritestack(clean_data{1}, clean_path)

            for jsigma = 1:numel(sigma_ratio_list)
                metadata("sr")="sr"+num2str(sigma_ratio_list(jsigma));
                sigma_ratio = sigma_ratio_list(jsigma);

                threshold = quantile(raw_data(:), 0.997);
                sigma = sigma_ratio * threshold;  % 设定噪声标准差（可调）
                noisy_data = add_rician_noise(raw_data, sigma);
                selected_data = extract_slice_from_3d(noisy_data, [slice_idx], direction);  % 使用默认轴向切片
                selected_data{1} = normalize_uint8(selected_data{1});

                noisy_join =  [metadata("modal"),metadata("type"),metadata("thickness"),metadata("pn"),metadata("rf"), ...
                    metadata("direction"),num2str(slice_idx),metadata("sr")];
                noisy_path = fullfile(noisy_folder, strjoin(noisy_join,"_")+".tif");
                imwritestack(selected_data{1}, noisy_path)

                noisy_stack_join =  [metadata("modal"),metadata("type"),metadata("thickness"),metadata("pn"),metadata("rf"),metadata("sr")];
                noisy_stack_path = fullfile(noisy_folder, strjoin(noisy_stack_join,"_")+".tif");
                imwritestack(noisy_data, noisy_stack_path)

                for jsearch = 1:numel(search_size_list)
                    metadata("search")=num2str(search_size_list(jsearch));
                    for jpatch = 1:numel(patch_size_list)
                        metadata("patch")=num2str(patch_size_list(jpatch));
                        for jh = 1:numel(h_list)
                            metadata("h")=num2str(h_list(jh));
                            % disp(metadata)
                            
                            k=k+1;
                            str = sprintf('Processing... %.2f%%', (k / total_progress) * 100);
                            waitbar(k/total_progress, progress_bar, str)

                            [denoised_img, psnr_value, ssim_value]=benchmark_2d(clean_data, selected_data,search_size_list(jsearch), ...
                                patch_size_list(jpatch), h_list(jh));

                            denoised_join = [metadata("modal"),metadata("type"),metadata("thickness"),metadata("pn"),metadata("rf"), ...
                                metadata("direction"),num2str(slice_idx),metadata("sr"),metadata("search"),metadata("patch"),metadata("h")];
                            denoised_path = fullfile(denoised_folder, strjoin(denoised_join, "_")+".tif");
                            imwritestack(denoised_img, denoised_path)

                            newRow = table(metadata("modal"),metadata("type"),metadata("thickness"),metadata("pn"),metadata("rf"), ...
                                metadata("direction"),string(slice_idx),metadata("sr"),metadata("search"),metadata("patch"),metadata("h"),psnr_value,ssim_value, ...
                                'VariableNames', {'Modal', 'Type', 'Thickness', 'PN', 'RF', 'Direction', ...
                                'Slice_Index', 'SR', 'Search', 'Patch', 'H', 'PSNR', 'SSIM'});

                            % 追加写入 CSV（如果文件不存在，会自动创建）
                            writetable(newRow, csv_path, 'Encoding', 'UTF-8', 'WriteMode', 'append');
                        end
                    end
                end
            end
        end
    end
end
delete(progress_bar);

function metadata = extract_metadata(file_name)
file_name = erase(file_name, ".tif"); % 去掉后缀
parts = split(file_name, "_");
metadata = dictionary('modal', parts(1));
if parts(2)=="icbm"
    metadata("type")="normal";
elseif parts(2)=="ai"
    metadata("type")="lesion";
else
    error("unexpected type. expected icbm_normal or ai_msles2")
end
metadata("thickness")=parts(4);
metadata("pn")=parts(5);
metadata("rf")=parts(6);
end