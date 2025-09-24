cfg = struct();
cfg.order            = 'avg_then_denoise'; % 或 'avg_then_denoise'denoise_then_avg
cfg.group_size       = 5;
cfg.denoise_method   = 'gaussian';           % 'none'|'median'|'gaussian'|'wiener'
cfg.denoise_strength = 0;                  % 0=关闭，1/2/3=轻/中/强
cfg.handle_tail      = 'keep_partial';     % 尾组不足5张：'drop'或'keep_partial'
cfg.save_format      = 'bmp';              % 'bmp' 或 'png'
cfg.strict_mode      = true;               % 严格要求 1024×1024×8bit

in_dir  = 'C:\Users\TomatoK\Desktop\20250911-coal&gangue\gangue\BMP\grey\1';         % 放置 image_1_0000.bmp 等
out_dir = 'C:\Users\TomatoK\Desktop\20250911-coal&gangue\gangue\BMP\ave';        % 程序会自动创建

batch_denoise_avg(in_dir, out_dir, cfg);
