function summary = avg_spectra_by5(in_path_or_list, out_dir, cfg)
% AVG_SPECTRA_BY5  将CSV光谱(列1=波长,列2..N=单次LIBS强度)按每5条平均成1条。
% 支持可选去噪(先/后次序可选)，一次性批量处理 18 个CSV文件。
%
% 用法示例：
%   cfg = struct();
%   cfg.group_size       = 5;                 % 每几条平均一条
%   cfg.handle_tail      = 'keep_partial';    % 'drop' 或 'keep_partial'
%   cfg.order            = 'denoise_then_avg';% 或 'avg_then_denoise'
%   cfg.denoise_method   = 'none';            % 'none'|'sgolay'|'median'
%   cfg.denoise_strength = 1;                 % 0/1/2/3 (0=关闭)
%   cfg.pattern          = '*.csv';           % 当 in_path_or_list 是文件夹时使用
%   cfg.split_groups     = false;             % true: 每个分组输出一个CSV；false: 单CSV汇总
%   cfg.output_prefix    = 'avg5_';           % 输出文件名前缀
%   cfg.strict_same_x    = false;             % 多文件时是否强制相同波长轴
%
%   % 方式A：给一个目录，批量处理其中18个CSV
%   summary = avg_spectra_by5('D:/spectra_folder', 'D:/out', cfg);
%
%   % 方式B：给18个CSV的全路径列表
%   files = { 'D:/in/A.csv','D:/in/B.csv', ... };
%   summary = avg_spectra_by5(files, 'D:/out', cfg);
%
% 输出：summary (struct array) 记录每个输入文件的处理信息。
%
% 作者：ChatGPT（GPT-5 Thinking），2025-09-18

if nargin < 1 || isempty(in_path_or_list), error('必须提供输入路径或文件列表'); end
if nargin < 2 || isempty(out_dir), error('必须提供输出目录 out_dir'); end
if nargin < 3, cfg = struct(); end
cfg = apply_defaults(cfg, struct( ...
    'group_size',5, ...
    'handle_tail','keep_partial', ...
    'order','denoise_then_avg', ...
    'denoise_method','none', ...
    'denoise_strength',1, ...
    'pattern','*.csv', ...
    'split_groups',false, ...
    'output_prefix','avg5_', ...
    'strict_same_x',false ...
));

if ~exist(out_dir,'dir'), mkdir(out_dir); end

% —— 归一化输入文件列表 ——
file_list = normalize_to_files(in_path_or_list, cfg.pattern);
if isempty(file_list)
    error('未在指定路径下找到任何CSV文件。');
end

summary = repmat(struct('file_in','', 'file_out','', 'n_shots',0, ...
                        'group_size',cfg.group_size, 'expected_groups',0, ...
                        'status','', 'message',''), 0, 1);

% —— 可选：收集并校验所有文件的波长轴一致性 ——
if cfg.strict_same_x
    ref_x = [];
end

for i = 1:numel(file_list)
    f = file_list{i};
    [folder, base, ~] = fileparts(f);
    try
        [x, Y] = read_spectrum_csv(f);
        nshots = size(Y,2);
        exp_groups = estimate_groups(nshots, cfg.group_size, cfg.handle_tail);

        if cfg.strict_same_x
            if isempty(ref_x)
                ref_x = x; %#ok<NASGU>
            else
                if ~isequaln(x, ref_x)
                    error('波长轴不一致（strict_same_x=true）。');
                end
            end
        end

        % —— 处理：去噪与分组平均 ——
        switch lower(cfg.order)
            case 'denoise_then_avg'
                Yd = apply_denoise_batch(Y, cfg);
                Z  = avg_columns_in_groups(Yd, cfg.group_size, cfg.handle_tail);
            case 'avg_then_denoise'
                Z0 = avg_columns_in_groups(Y, cfg.group_size, cfg.handle_tail);
                Z  = apply_denoise_batch(Z0, cfg);
            otherwise
                error('未知 order：%s', cfg.order);
        end

        % —— 导出 ——
        if cfg.split_groups
            % 多文件：每个分组保存一个CSV
            for g = 1:size(Z,2)
                outname = sprintf('%s%s_g%04d.csv', cfg.output_prefix, base, g);
                outpath = fullfile(out_dir, outname);
                write_spectrum_csv(outpath, x, Z(:,g), sprintf('avg_g%04d', g));
            end
            file_out = '(multiple files)';
        else
            % 单文件：一份CSV包含所有分组
            outname = sprintf('%s%s.csv', cfg.output_prefix, base);
            outpath = fullfile(out_dir, outname);
            varnames = arrayfun(@(k) sprintf('avg_g%04d',k), 1:size(Z,2), 'UniformOutput', false);
            write_spectrum_csv(outpath, x, Z, varnames);
            file_out = outpath;
        end

        S = struct('file_in',f,'file_out',file_out,'n_shots',nshots, ...
                   'group_size',cfg.group_size,'expected_groups',exp_groups,'status','ok','message','');
        summary(end+1,1) = S; %#ok<AGROW>
        fprintf('[OK] %s  shots=%d  → %s\n', f, nshots, file_out);
    catch ME
        S = struct('file_in',f,'file_out','', 'n_shots',NaN, ...
                   'group_size',cfg.group_size,'expected_groups',NaN,'status','error','message',ME.message);
        summary(end+1,1) = S; %#ok<AGROW>
        warning('[FAIL] %s\n  -> %s', f, ME.message);
    end
end

fprintf('[DONE] 完成处理 %d 个CSV。输出目录：%s\n', numel(file_list), out_dir);

end % ===== 主函数结束 =====

%% ========== 数据读取与导出 ==========
function [x, Y] = read_spectrum_csv(filepath)
% 读取一份CSV：第一列=波长，2..N=强度
M = readmatrix(filepath);
if size(M,2) < 2
    error('文件列数不足：%s', filepath);
end
x = M(:,1);
Y = M(:,2:end);
% 去除任何全NaN行
valid = ~all(isnan([x Y]),2);
x = x(valid);
Y = Y(valid,:);
end

function write_spectrum_csv(filepath, x, Z, varnames)
% 将一组或多组平均光谱写出到CSV
% Z: n x G 矩阵
if ischar(varnames) || isstring(varnames)
    varnames = {char(varnames)};
end
if size(Z,2) ~= numel(varnames)
    % 自动补齐变量名
    varnames = arrayfun(@(k) sprintf('avg_g%04d',k), 1:size(Z,2), 'UniformOutput', false);
end
header = ['wavelength', varnames];
C = [num2cell(x), num2cell(Z)];
writecell([header; C], filepath);
end

%% ========== 去噪与分组 ==========
function Yd = apply_denoise_batch(Y, cfg)
% 对每一列(一条光谱)按行方向(沿波长)去噪
if cfg.denoise_strength<=0 || strcmpi(cfg.denoise_method,'none')
    Yd = Y; return;
end
switch lower(cfg.denoise_method)
    case 'sgolay'
        [win, poly] = sgolay_params(cfg.denoise_strength);
        % 沿第1维滤波
        Yd = Y;
        for j = 1:size(Y,2)
            try
                Yd(:,j) = sgolayfilt(Y(:,j), poly, win);
            catch
                % 无信号处理工具箱则回退为移动平均
                Yd(:,j) = movmean(Y(:,j), max(3,win));
            end
        end
    case 'median'
        k = kernel_len(cfg.denoise_strength); % 3/5/7
        Yd = Y;
        for j = 1:size(Y,2)
            try
                Yd(:,j) = medfilt1(Y(:,j), k, 'omitnan', 'truncate');
            catch
                % 无 medfilt1 则回退为 moving median 近似
                Yd(:,j) = movmedian(Y(:,j), k, 'omitnan');
            end
        end
    otherwise
        warning('未知去噪方法：%s，已忽略。', cfg.denoise_method);
        Yd = Y;
end
end

function Z = avg_columns_in_groups(Y, g, handle_tail)
% 将列按 g 分组，并对每组取均值（忽略NaN）
N = size(Y,2);
cols = 1:N;
ng = ceil(N/g);
Z = [];
for i = 1:ng
    s = (i-1)*g + 1; e = min(i*g, N);
    if strcmpi(handle_tail,'drop') && (e-s+1 < g)
        break; % 丢弃尾组
    end
    Zi = mean(Y(:, s:e), 2, 'omitnan');
    Z = [Z, Zi]; %#ok<AGROW>
end
end

%% ========== 工具 ==========
function files = normalize_to_files(in_path_or_list, pattern)
if ischar(in_path_or_list) || isstring(in_path_or_list)
    p = char(in_path_or_list);
    if isfolder(p)
        L = dir(fullfile(p, pattern));
        files = cellfun(@(n) fullfile(L(1).folder, n), {L.name}, 'UniformOutput', false);
        % 稍作自然排序（按名字提取末尾数字）
        if ~isempty(files)
            files = natural_sort_by_trailing_number(files);
        end
    else
        files = {p};
    end
elseif iscell(in_path_or_list)
    files = in_path_or_list(:)';
else
    error('不支持的输入类型：%s', class(in_path_or_list));
end
end

function files_sorted = natural_sort_by_trailing_number(files)
idx = nan(numel(files),1);
for i = 1:numel(files)
    [~, base, ~] = fileparts(files{i});
    t = regexp(base, '(\d+)$', 'tokens', 'once');
    if ~isempty(t), idx(i) = str2double(t{1}); else, idx(i) = i; end
end
[~, order] = sort(idx);
files_sorted = files(order);
end

function n = estimate_groups(n_shots, g, handle_tail)
switch lower(handle_tail)
    case 'drop', n = floor(n_shots/g);
    otherwise,   n = ceil(n_shots/g);
end
end

function k = kernel_len(strength)
% 1/2/3 → 3/5/7
k = 2*max(1,min(3,round(strength))) + 1;
end

function [win, poly] = sgolay_params(strength)
% 1/2/3 → window/polynomial order
switch max(1,min(3,round(strength)))
    case 1, win = 7;  poly = 2;
    case 2, win = 11; poly = 3;
    case 3, win = 17; poly = 3;
end
end

function cfg = apply_defaults(cfg, defs)
fn = fieldnames(defs);
for i=1:numel(fn)
    k = fn{i};
    if ~isfield(cfg,k) || isempty(cfg.(k))
        cfg.(k) = defs.(k);
    end
end
end
