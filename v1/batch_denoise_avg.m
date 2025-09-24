ifunction summary = batch_denoise_avg(in_dirs, out_root, cfg)
% BATCH_DENOISE_AVG  批量读取 BMP（1024x1024x8bit 灰度），可选先去噪/后去噪，
% 每 5 幅逐像素平均并导出到指定目录。现支持：
%   1) 单一文件夹；
%   2) 多个文件夹（传入 cellstr）；
%   3) 根目录 + 自动枚举子文件夹（cfg.scan_subfolders=true）。
%
% 用法（单文件夹，保持兼容）：
%   cfg = struct();
%   cfg.order            = 'denoise_then_avg';  % 或 'avg_then_denoise'
%   cfg.group_size       = 5;                   % 每组张数
%   cfg.denoise_method   = 'median';            % 'none'|'median'|'gaussian'|'wiener'
%   cfg.denoise_strength = 1;                   % 0|1|2|3 （0 关闭）
%   cfg.handle_tail      = 'keep_partial';      % 'drop'|'keep_partial'
%   cfg.strict_mode      = true;                % 严格校验尺寸/类型
%   cfg.save_format      = 'bmp';               % 'bmp'|'png'
%   cfg.expected_size    = [1024 1024];         % 期望尺寸
%   cfg.name_pattern     = 'image_1_*.bmp';     % 读取文件名模式
%   % —— 新增 ——
%   cfg.scan_subfolders  = false;               % 若为 true 且 in_dirs 为单一路径，则枚举其下一层子文件夹
%   cfg.out_subdir       = true;                % 多文件夹时，输出按输入子文件夹名分开
%   cfg.out_name_pattern = 'avg_{start}-{end}.{ext}'; % 自定义输出命名（支持占位符）
%   % 占位符：{folder} {start} {end} {g} {ext}
%   % 也可用函数：cfg.out_name_func = @(folder, s, e, g, nd, ext) sprintf('avg_%s_%0*d-%0*d.%s', folder, nd, s, nd, e, ext);
%   
%   summary = batch_denoise_avg('D:/input_folder', 'D:/output_folder', cfg);
%
% 用法（18 个子文件夹）：
%   folders = { 'D:/data/F01', 'D:/data/F02', ..., 'D:/data/F18' };
%   summary = batch_denoise_avg(folders, 'D:/output_root', cfg);
%
% 用法（根目录 + 自动枚举子目录）：
%   cfg.scan_subfolders = true;
%   summary = batch_denoise_avg('D:/root_with_18_folders', 'D:/output_root', cfg);
%
% 返回：summary（结构体数组），记录每个输入文件夹的处理信息。
%
% 作者：ChatGPT（GPT-5 Thinking），2025-09-18

% —— 参数与默认值 ——
if nargin < 1 || isempty(in_dirs), error('必须提供输入目录或目录列表 in_dirs'); end
if nargin < 2 || isempty(out_root), error('必须提供输出目录 out_root'); end
if nargin < 3, cfg = struct(); end
cfg = apply_defaults(cfg, struct( ...
    'order','denoise_then_avg', ...
    'group_size',5, ...
    'denoise_method','median', ...
    'denoise_strength',1, ...
    'handle_tail','keep_partial', ...
    'strict_mode',true, ...
    'save_format','bmp', ...
    'show_preview',false, ...
    'expected_size',[1024 1024], ...
    'name_pattern','image_1_*.bmp', ...
    'scan_subfolders',false, ...
    'out_subdir',true, ...
    'out_name_pattern','avg_{start}-{end}.{ext}', ...
    'gdigits',4 ...
));

if ~exist(out_root,'dir'), mkdir(out_root); end

% —— 归一化输入：得到需要处理的文件夹列表 ——
folders = normalize_input_to_folders(in_dirs, cfg);
if isempty(folders)
    error('未发现可处理的文件夹。');
end

fprintf('[BATCH] 计划处理文件夹数：%d\n', numel(folders));

summary = repmat(struct('folder_in','', 'folder_out','', 'n_files',0, ...
                        'group_size',cfg.group_size, 'expected_groups',0, ...
                        'status','', 'message',''), 0, 1);

for i = 1:numel(folders)
    in_dir = folders{i};
    if endsWith(in_dir, filesep), in_dir = in_dir(1:end-1); end
    [~, subname] = fileparts(in_dir);
    if cfg.out_subdir
        out_dir = fullfile(out_root, subname);
    else
        out_dir = out_root;
    end
    if ~exist(out_dir,'dir'), mkdir(out_dir); end

    % 预统计 BMP 数，用于日志
    L = dir(fullfile(in_dir, cfg.name_pattern));
    n_files = numel(L);
    if n_files == 0
        fprintf('[SKIP] %s  —— 无匹配文件：%s\n', in_dir, cfg.name_pattern);
        S = struct('folder_in',in_dir,'folder_out',out_dir,'n_files',0,'group_size',cfg.group_size, ...
                   'expected_groups',0,'status','skipped','message','no files');
        summary(end+1,1) = S; %#ok<AGROW>
        continue;
    end
    exp_groups = estimate_groups(n_files, cfg.group_size, cfg.handle_tail);
    fprintf('[RUN] (%d/%d) %s | files=%d | → %s\n', i, numel(folders), in_dir, n_files, out_dir);

    try
        % —— 实际处理一个文件夹 ——
        gcount = process_one_folder(in_dir, out_dir, subname, cfg);
        status = 'ok'; msg = '';
    catch ME
        gcount = 0; status = 'error'; msg = ME.message;
        warning('[FAIL] %s\n  -> %s', in_dir, msg);
    end

    S = struct('folder_in',in_dir,'folder_out',out_dir,'n_files',n_files,'group_size',cfg.group_size, ...
               'expected_groups',exp_groups,'status',status,'message',msg);
    summary(end+1,1) = S; %#ok<AGROW>
end

fprintf('[DONE] 所有文件夹处理完成：%d 个。输出根目录：%s\n', numel(folders), out_root);

end % ===== 主函数结束 =====

%% ========== 子例程：处理一个文件夹 ==========
function G = process_one_folder(in_dir, out_dir, folder_name, cfg)
% 列出并排序文件
L = dir(fullfile(in_dir, cfg.name_pattern));
if isempty(L)
    error('在 %s 下未找到符合模式 %s 的 BMP 文件。', in_dir, cfg.name_pattern);
end
[files_sorted, idx_sorted, ndigits] = sort_by_trailing_number(L);
N = numel(files_sorted);

fprintf('  [Info] %s  Found %d files. Group size = %d. Order = %s. Denoise=%s(%d).\n', ...
    folder_name, N, cfg.group_size, cfg.order, cfg.denoise_method, cfg.denoise_strength);

G = 0; s = 1; % 输出组计数与游标
while s <= N
    e = min(s + cfg.group_size - 1, N);
    group_len = e - s + 1;
    if strcmpi(cfg.handle_tail,'drop') && group_len < cfg.group_size
        fprintf('  [Warn] Dropping tail group [%d..%d] (size=%d).\n', s, e, group_len);
        break;
    end

    % 读入并处理当前组
    acc = [];  raw_first = [];  den_first = [];
    for k = s:e
        f = files_sorted(k);
        fpath = fullfile(f.folder, f.name);
        Iu8 = imread(fpath);
        I = normalize_to_gray(Iu8, cfg, fpath);
        if strcmpi(cfg.order,'denoise_then_avg')
            J = apply_denoise(I, cfg.denoise_method, cfg.denoise_strength);
        else
            J = I; % avg_then_denoise：先累加原图，之后再去噪
        end
        if isempty(acc)
            acc = zeros(size(J),'double');
        end
        acc = acc + im2double(J);
        if cfg.show_preview && isempty(raw_first)
            raw_first = I; den_first = J;
        end
    end
    avg = acc ./ group_len; % double in [0,1]

    if strcmpi(cfg.order,'avg_then_denoise')
        avg = apply_denoise(avg, cfg.denoise_method, cfg.denoise_strength);
    end

    % 预览
    if cfg.show_preview
        try
            figure('Name',sprintf('%s | Group %d preview',folder_name,G+1));
            if strcmpi(cfg.order,'denoise_then_avg')
                subplot(1,3,1); imshow(raw_first,[]); title('Raw sample');
                subplot(1,3,2); imshow(den_first,[]); title('Denoised sample');
                subplot(1,3,3); imshow(avg,[]); title('Group average');
            else
                subplot(1,2,1); imshow(raw_first,[]); title('Raw sample');
                subplot(1,2,2); imshow(avg,[]); title('Group average → (then denoise)');
            end
            drawnow;
        catch
        end
    end

    % —— 保存输出（自定义命名） ——
    G = G + 1;
    idx_start = idx_sorted(s); idx_end = idx_sorted(e);
    outname = make_outname(cfg, folder_name, idx_start, idx_end, G, ndigits);
    outpath = fullfile(out_dir, outname);

    outu8 = im2uint8(min(max(avg,0),1));
    imwrite(outu8, outpath);
    fprintf('  [Save] %s  (from %s .. %s)\n', outpath, files_sorted(s).name, files_sorted(e).name);

    s = e + 1;
end

fprintf('  [Done] %s  Output groups: %d → %s\n', folder_name, G, out_dir);

end

%% =========== 工具函数（公用） ===========
function cfg = apply_defaults(cfg, defs)
fn = fieldnames(defs);
for i=1:numel(fn)
    k = fn{i};
    if ~isfield(cfg,k) || isempty(cfg.(k))
        cfg.(k) = defs.(k);
    end
end
end

function folders = normalize_input_to_folders(in_dirs, cfg)
% 将输入统一为文件夹 cell 数组
if ischar(in_dirs) || isstring(in_dirs)
    p = char(in_dirs);
    if cfg.scan_subfolders
        % 枚举下一层子目录（含根目录自身若有匹配文件）
        folders = enumerate_candidate_folders(p, cfg.name_pattern);
    else
        folders = {p};
    end
elseif iscell(in_dirs)
    % 过滤不存在路径
    ok = cellfun(@isfolder, in_dirs);
    if any(~ok)
        warning('以下文件夹不存在，已跳过：\n%s', strjoin(in_dirs(~ok), '\n'));
    end
    folders = in_dirs(ok);
else
    error('in_dirs 类型不支持：%s', class(in_dirs));
end
end

function subfolders = enumerate_candidate_folders(root_dir, name_pattern)
if ~isfolder(root_dir), error('根目录不存在：%s', root_dir); end
D = dir(root_dir); D = D([D.isdir]);
D = D(~ismember({D.name},{'.','..'}));
subfolders = {};
for k = 1:numel(D)
    p = fullfile(D(k).folder, D(k).name);
    if ~isempty(dir(fullfile(p, name_pattern)))
        subfolders{end+1} = p; %#ok<AGROW>
    end
end
% 根目录本身若含匹配文件，也一并纳入
if ~isempty(dir(fullfile(root_dir, name_pattern)))
    subfolders = [{root_dir}, subfolders];
end
end

function [files_sorted, idx_sorted, ndigits] = sort_by_trailing_number(L)
% 按文件名最后的连续数字排序；返回排序后的列表、对应数字索引（double），以及数字位数
names = {L.name};
idx = nan(numel(names),1);
ndigits = 1;
for i=1:numel(names)
    t = regexp(names{i}, '(\d+)(?=\.[^\.]+$)', 'tokens', 'once');
    if ~isempty(t)
        idx(i) = str2double(t{1});
        ndigits = max(ndigits, numel(t{1}));
    else
        % 若找不到尾随数字，则退而求其次：提取任意数字序列的最后一个
        t2 = regexp(names{i}, '(\d+)', 'tokens');
        if ~isempty(t2)
            idx(i) = str2double(t2{end}{1});
            ndigits = max(ndigits, numel(t2{end}{1}));
        else
            idx(i) = inf; % 放到最后
        end
    end
end
[~, order] = sort(idx);
files_sorted = L(order);
idx_sorted   = idx(order);
if isinf(idx_sorted(end))
    warning('部分文件未能解析出数字索引，将排在末尾。');
end
end

function I = normalize_to_gray(Iu8, cfg, fpath)
% 统一转换到 double[0,1] 的灰度图像（尺寸校验/调整）
if ndims(Iu8) == 3
    Iu8 = rgb2gray(Iu8);
end
if ~isa(Iu8,'uint8')
    if cfg.strict_mode
        error('输入图像非 uint8：%s', fpath);
    else
        warning('非 uint8 图像，已转换：%s', fpath);
        Iu8 = im2uint8(mat2gray(Iu8));
    end
end
[h,w] = size(Iu8);
if any([h,w] ~= cfg.expected_size)
    if cfg.strict_mode
        error('尺寸不匹配（%dx%d 期望 %dx%d）：%s', h,w,cfg.expected_size(1),cfg.expected_size(2), fpath);
    else
        warning('尺寸不匹配，已 resize 到期望尺寸：%s', fpath);
        Iu8 = imresize(Iu8, cfg.expected_size, 'bilinear');
    end
end
I = im2double(Iu8);
end

function J = apply_denoise(I, method, strength)
% I: double[0,1] 灰度
if strength<=0 || strcmpi(method,'none')
    J = I; return;
end
switch lower(method)
    case 'median'
        k = pick_kernel(strength); % 3/5/7
        J = medfilt2(I, [k k], 'symmetric');
    case 'gaussian'
        [sigma, k] = pick_gauss(strength); %#ok<ASGLU>
        % 优先使用 imgaussfilt，如不可用退回 imfilter
        try
            J = imgaussfilt(I, sigma, 'FilterSize', k, 'Padding','symmetric');
        catch
            h = fspecial('gaussian', k, sigma);
            J = imfilter(I, h, 'symmetric');
        end
    case 'wiener'
        k = pick_kernel(strength);
        try
            J = wiener2(I, [k k]);
        catch
            % 若无 wiener2，退化为中值
            J = medfilt2(I, [k k], 'symmetric');
        end
    otherwise
        warning('未知去噪方法：%s，已忽略。', method);
        J = I;
end
end

function k = pick_kernel(strength)
% 将 1/2/3 → 3/5/7
k = 2*strength + 1; % 1→3, 2→5, 3→7
end

function [sigma,k] = pick_gauss(strength)
% 经验映射：1→σ0.6, 2→σ1.0, 3→σ1.6
sigma_table = [0.6, 1.0, 1.6];
strength = max(1, min(3, round(strength)));
sigma = sigma_table(strength);
k = 2*ceil(2*sigma)+1; % ~覆盖±2σ
end

function n = estimate_groups(n_files, group_size, handle_tail)
switch lower(handle_tail)
    case 'drop', n = floor(n_files/group_size);
    otherwise,   n = ceil(n_files/group_size);
end
end

function name = make_outname(cfg, folderName, idxStart, idxEnd, gidx, ndigits)
% 根据配置生成输出文件名
ext = lower(cfg.save_format);
if isfield(cfg,'out_name_func') && isa(cfg.out_name_func,'function_handle')
    name = cfg.out_name_func(folderName, idxStart, idxEnd, gidx, ndigits, ext);
    return;
end
% 默认使用简单占位符替换
pattern = getfield_or(cfg,'out_name_pattern','avg_{start}-{end}.{ext}');
start_str = num2str_pad(idxStart, ndigits);
end_str   = num2str_pad(idxEnd,   ndigits);
Gdigits   = getfield_or(cfg,'gdigits', max(4, ndigits));
g_str     = num2str_pad(gidx, Gdigits);
name = pattern;
name = strrep(name,'{folder}', folderName);
name = strrep(name,'{start}',  start_str);
name = strrep(name,'{end}',    end_str);
name = strrep(name,'{g}',      g_str);
name = strrep(name,'{ext}',    ext);
end

function s = num2str_pad(n, nd)
s = sprintf(['%0', num2str(nd), 'd'], n);
end

function v = getfield_or(S, k, vdef)
if isstruct(S) && isfield(S,k) && ~isempty(S.(k))
    v = S.(k);
else
    v = vdef;
end
end
