clear; clc; close all;
addpath(pwd);  % 确保 avg_spectra_by5.m 在路径上

cfg = struct();
cfg.group_size       = 5;                 % 每5条平均
cfg.handle_tail      = 'keep_partial';    % 尾组不足5条：保留或用 'drop' 丢弃
cfg.order            = 'denoise_then_avg';% 顺序随意（反正不去噪）
cfg.denoise_method   = 'none';            % 关闭去噪
cfg.denoise_strength = 0;                 % 关闭去噪
cfg.split_groups     = false;             % 输出到一个CSV中（每组一列）
cfg.output_prefix    = 'avg5_';
cfg.pattern          = '*.csv';

in_dir  = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\origin';  % 里面有18个CSV
out_dir = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\ave_5';

summary = avg_spectra_by5(in_dir, out_dir, cfg);
