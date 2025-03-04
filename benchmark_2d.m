function [denoised_img, psnr_value, ssim_value] = benchmark_2d(clean_data, selected_data, search_size, patch_size, h)
% 此函数只支持单张图片的fast NLM

tic;  % 计时
% disp('正在使用 fast_NLM 进行去噪...');
denoised_data = fast_nlm_gpu(selected_data, h, search_size, patch_size);


if iscell(denoised_data)
    for i = 1 : numel(denoised_data)
        denoised_data{i} = normalize_uint8(denoised_data{i});
        % denoised_data{i} = uint8(denoised_data{i});
    end
else
    denoised_data = normalize_uint8(denoised_data);
end
denoising_time = toc;
% disp(['去噪完成！耗时: ', num2str(denoising_time), ' 秒']);

psnr_value = psnr(denoised_data{1}, clean_data{1});
ssim_value = ssim(denoised_data{1}, clean_data{1});

% disp(['PSNR: ', num2str(psnr_value)]);
% disp(['SSIM: ', num2str(ssim_value)]);

denoised_img = denoised_data{1};
end