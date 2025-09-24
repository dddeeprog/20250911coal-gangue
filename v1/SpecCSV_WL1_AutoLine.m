%% 0) 清空环境 + 路径
clc; clear; close all;
cd 'C:\Users\TomatoK\Desktop\20250911coal&gangue\code_v1\Matlab\v1';
addpath(genpath('./FUN'));   % 里边有 baseline_arPLS / autolineV4 / DataSlec / plotstyle 等

%% 1) 读取当前数据格式（目录下有若干 CSV；每个 CSV: 第一列=波长，其余列=该样品的多幅光谱）
% 例：C:\...\targets\ 下面有 1.csv、2.csv、...、6.csv
DATA_DIR = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\all';
OUT_DIR  = fullfile(DATA_DIR,'re');
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

files = dir(fullfile(DATA_DIR,'*.csv'));
assert(~isempty(files), '目录中未找到 *.csv 文件：%s', DATA_DIR);

% 按文件名里的数字排序（没有数字则按名字）
nums = nan(numel(files),1);
for i = 1:numel(files)
    t = regexp(files(i).name,'(\d+)','tokens','once');
    if ~isempty(t), nums(i) = str2double(t{1}); end
end
if any(~isnan(nums))
    [~,ord] = sort(nums);
    files   = files(ord);
else
    [~,ord] = sort({files.name});
    files   = files(ord);
end

% 读取首个文件确定公共波长轴
A1  = readmatrix(fullfile(files(1).folder, files(1).name));
wave = A1(:,1);                      % M×1
M    = numel(wave);

% 堆叠所有样品的复测（把每个 CSV 的列转成行，再纵向堆叠）
dataset    = [];                     % [总复测数 K_all] × [波长数 M]
sample_idx = [];                     % 每条复测对应的“样品序号”（1..Nsamples）
rep_idx    = [];                     % 每条复测在该样品内的序号（1..Ki）
sample_names = strings(numel(files),1);

for s = 1:numel(files)
    fi = fullfile(files(s).folder, files(s).name);
    T  = readmatrix(fi);
    assert(size(T,1)==M, '文件 %s 的波长点数(%d)与首个文件(%d)不一致。', files(s).name, size(T,1), M);
    if max(abs(T(:,1)-wave)) > 1e-6*max(1,max(abs(wave)))
        warning('文件 %s 的波长轴与首个文件略有差异，请确认是否需要重采样/对齐。', files(s).name);
        % 如需严格对齐，可在此处做插值到 wave 轴
    end
    Xs = T(:,2:end);                 % M×Ki
    dataset    = [dataset; Xs.'];    %#ok<AGROW>  % Ki×M
    sample_idx = [sample_idx; repmat(s, size(Xs,2), 1)]; %#ok<AGROW>
    rep_idx    = [rep_idx; (1:size(Xs,2)).'];             %#ok<AGROW>
    sample_names(s) = string(files(s).name);
end

% 现在：wave(M×1)，dataset(K_all×M)，每行是一幅光谱
fprintf('[Info] 读取完成：样品数=%d，总复测数=%d，波长点=%d\n', numel(files), size(dataset,1), M);

%% 2) 光谱展示 + 自动/手动选线（与你原逻辑一致）
datashow = dataset;                   % [复测 × 波长]
eleslec_names = {'C','H','N','O','Si','Al','Fe'};  % 机选的元素名（可改）

% —— 求平均光谱并去基线
line = mean(datashow, 1);            % 1×M（行向量）
[Spectrum1,~] = baseline_arPLS(line);% 返回向量（保持行向量）
wave_row = wave(:).';                % 1×M
spec_row = Spectrum1(:).';           % 1×M
assert(numel(spec_row)==numel(wave_row), 'wave 与光谱长度不一致');

% —— 画平均谱（去基线后）
figure; clf; 
plot(wave, Spectrum1);
plotstyle('x','Wavelength (nm)','y','Intensity (a.u.)','ptitle','Mean spectrum (baseline removed)');
set(gcf,'position',[100,100,900,300]);
xlim([min(wave) max(wave)]);
ylim([min(Spectrum1)-0.05*range(Spectrum1), max(Spectrum1)+0.05*range(Spectrum1)]);

% —— 自动找线（机选）
[peaks_find, elemnames] = autolineV4(spec_row, wave_row, eleslec_names);

% —— 导出机选结果与平均谱
writecell(elemnames, fullfile(OUT_DIR,'waveslec_auto.xlsx'), 'WriteMode','overwrite');
writematrix(peaks_find, fullfile(OUT_DIR,'waveslec_auto.xlsx'), 'Sheet',1,'Range','B1');
linewrite = [wave(:), Spectrum1(:)];
writematrix(linewrite, fullfile(OUT_DIR,'mean_spectrum_baseline.csv'), 'WriteMode','overwrite');

%% 3) （可选）读取“人工确认/手改”的线表，并抽取对应光谱列做快速浏览
% 期望格式：第一列是线名，其余列是波长；例如你手动编辑后存为 waveslec1.xlsx
xls_manual = fullfile(OUT_DIR,'waveslec1.xlsx');
if exist(xls_manual,'file')
    waveslec      = readcell(xls_manual, 'Sheet', 1);
    waveslec_name = waveslec(:,1);
    waveslec_num  = cell2mat(waveslec(:,2:end));
    % 这里取第6列作为最终波长（保持你原来的习惯；如需其他列，改这个索引即可）
    col_pick      = min(6, size(waveslec_num,2));
    wave_slec     = waveslec_num(:, col_pick);

    % 抽取这些波长附近的强度（DataSlec: 你 FUN 里的函数，参数保持原来写法）
    datashow_sel = DataSlec(dataset, wave, 1, 1, wave_slec);   % [复测 × 选线数]

    % 快速查看前若干复测在这些线位的强度变化
    n_show_spec = min(100, size(datashow_sel,1));
    n_lines     = size(datashow_sel,2);
    x = (1:n_show_spec).';
    for i = 1:n_lines
        figure('Name', sprintf('Line #%d  (%.4f nm)', i, wave_slec(i)));
        plot(x, datashow_sel(1:n_show_spec, i), '-');
        xlabel('spectrum index'); ylabel('intensity (a.u.)');
        title(sprintf('Selected line %d @ %.4f nm', i, wave_slec(i)));
        grid on;
    end
else
    fprintf('[Hint] 未检测到人工线表：%s\n', xls_manual);
    fprintf('      你可以先打开 %s 手动修改/确认，然后重新运行这一段。\n', fullfile(OUT_DIR,'waveslec_auto.xlsx'));
end
