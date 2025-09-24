%% CSVs → 单一 MAT（wavelengths, spec1, spec2, ...）
% 读取目录下的若干 CSV（第1列=光谱波长，其余列=光谱强度或已平均结果），
% 依次生成变量 wavelengths（仅取第一个CSV的波长列）、spec1、spec2、...，
% 并保存为一个 MAT（-v7.3）。不做任何去噪/再次平均。

clear; clc;

%% === 可配置区 ===
% 输入目录（放 18 个 CSV）
in_dir   = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\ave_5\all';     % ← 修改为你的CSV目录
pattern  = '*.csv';                  % 枚举通配符
% 输出 MAT 文件路径
out_mat  = fullfile(in_dir, 'all.mat');

% 行/列处理
skip_header     = true;  % 第一行是标签：去掉表头
drop_nan_rows   = true;  % 丢弃全NaN行
force_same_axis = false; % 若 true：强制所有CSV的第一列与首个一致，否则报错

%% === 枚举并排序 CSV 列表（按文件名末尾数字自然排序；无数字则按出现顺序） ===
L = dir(fullfile(in_dir, pattern));
if isempty(L)
    error('在 %s 下未找到 CSV（模式：%s）。', in_dir, pattern);
end
files = cell(numel(L),1);
keys_hasnum = false(numel(L),1);
keys_num    = zeros(numel(L),1);
for i = 1:numel(L)
    files{i} = fullfile(L(i).folder, L(i).name);
    [~, base, ~] = fileparts(L(i).name);
    t = regexp(base, '(\d+)$', 'tokens', 'once');
    if ~isempty(t)
        keys_hasnum(i) = true;
        keys_num(i)    = str2double(t{1});
    else
        keys_hasnum(i) = false;
        keys_num(i)    = i; % 保留原顺序
    end
end
T = [~keys_hasnum(:), keys_num(:)];   % 先有数字(0)再无数字(1)，再按数字升序
[~, order] = sortrows(T, [1 2]);
files = files(order);

%% === 读取与装载到 wavelengths / spec1 / spec2 / ... 变量 ===
spec_idx = 1;
varnames = {};
wavelengths = [];
for i = 1:numel(files)
    f = files{i};
    try
        M = readmatrix(f);
        if isempty(M) || size(M,2) < 1
            warning('[SKIP] %s  列数不足。', f);
            continue;
        end
        % --- 去掉第一行表头（若有） ---
        if skip_header && size(M,1) >= 1
            M = M(2:end, :);
        end
        % --- 丢弃全NaN行（兼容表头/空行/分隔符） ---
        if drop_nan_rows
            valid = ~all(isnan(M),2);
            M = M(valid,:);
        end
        if size(M,2) < 2
            warning('[SKIP] %s  仅1列（无强度列），已跳过。', f);
            continue;
        end
        % --- 处理波长轴 ---
        x = M(:,1);
        if isempty(wavelengths)
            wavelengths = x; %#ok<NASGU>
        else
            if force_same_axis
                if ~isequaln(x, wavelengths)
                    error('文件 %s 的波长轴与首个不一致（force_same_axis=true）。', L(order(i)).name);
                end
            else
                if ~isequaln(size(x), size(wavelengths)) || any(~isfinite(x))
                    warning('[WARN] %s 的波长轴尺寸/数值可能与首个不一致。', L(order(i)).name);
                end
            end
        end
        % --- 强度矩阵（去掉第1列波长） ---
        S = M(:,2:end);
        vname = sprintf('spec%d', spec_idx);
        eval([vname ' = S;']); %#ok<EVLDIR>
        varnames{end+1} = vname; %#ok<SAGROW>
        spec_idx = spec_idx + 1;
        fprintf('[OK] %s  →  %-6s  wavelengths(%d)  S=(%dx%d)\n', L(order(i)).name, vname, numel(x), size(S,1), size(S,2));
    catch ME
        warning('[FAIL] %s\n  -> %s', f, ME.message);
    end
end

if isempty(varnames) || isempty(wavelengths)
    error('没有成功读取到任何 CSV 或未获得波长轴。');
end

%% === 保存到单一 MAT：wavelengths + spec1, spec2, ... ===
save(out_mat, 'wavelengths', varnames{:}, '-v7.3');
fprintf('\n[SAVED] %s  （wavelengths + %d 个spec变量：%s...）\n', out_mat, numel(varnames), strjoin(varnames(1:min(3,end)), ', '));

%% 完成
