%% PLSR（多指标；训练折外 + 外测留出；叠加误差棒）+ VIP + T²/Q
% 蓝色：训练样品的样品级折外预测（均值±STD）
% 红色：外测样品的样品级预测（均值±STD）
% 1:1 参考线；标题显示 R² / RMSE（训练折外与外测分别统计）
clear; clc; close all; rng(2025,'twister');

%% ===== 路径与输出 =====
% 煤矸石
ave_mat     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\all\all.mat';
excel_xlsx  = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\all\targets\targets_18.xlsx';
excel_sheet = '';   % 指定表名或留空自动
out_dir     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\all\plsr_out_holdout18';
if ~exist(out_dir,'dir'), mkdir(out_dir); end
% % 煤
% ave_mat     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\ave_5\all\all.mat';
% excel_xlsx  = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\ave_5\all\targets\targets_18.xlsx';
% excel_sheet = '';   % 指定表名或留空自动
% out_dir     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\coal\spec\ave_5\all\plsr_out_holdout18';
% if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 配置：固定 N 个样品 + 外测留出策略 =====
% 煤矸石
FORCE_N        = 18;      % 使用前 N 个样品（按变量名尾号排序）
holdout_list   = [3 4 9 10 15 16];   % 外测样品索引（基于排序后的 1..FORCE_N）
holdout_k      = 4;       % 当 holdout_list 为空时，随机留出的样品数
holdout_ratio  = [];      % 或者按比例（如 0.2）
rng_seed       = 2025;    % 随机种子
rng(rng_seed,'twister');
% % 煤
% FORCE_N        = 13;      % 使用前 N 个样品（按变量名尾号排序）
% holdout_list   = [3 4 8 12];   % 外测样品索引（基于排序后的 1..FORCE_N）
% holdout_k      = 4;       % 当 holdout_list 为空时，随机留出的样品数
% holdout_ratio  = [];      % 或者按比例（如 0.2）
% rng_seed       = 2025;    % 随机种子
% rng(rng_seed,'twister');

%% ===== 指标与同义列名 =====
response_names = {'TotalS','Ash','Volatile','HHV','C','H','N'};
synonyms = containers.Map();
synonyms('TotalS')  = lower({'TotalS','S','Sulfur','Total_S','S_total'});
synonyms('Ash')     = lower({'Ash','AshContent','Ash_Content'});
synonyms('Volatile')= lower({'Volatile','VM','Vol','VolatileMatter'});
synonyms('HHV')     = lower({'HHV','GCV','CalorificValue','HighHeatingValue'});
synonyms('C')       = lower({'C','Carbon'});
synonyms('H')       = lower({'H','Hydrogen'});
synonyms('N')       = lower({'N','Nitrogen'});

%% ===== 读取 MAT（识别波长与样品矩阵） =====
S = load(ave_mat); fn = fieldnames(S);
axis_candidates = {'wavelengths','wavelength','lambda','wavelenths'};
wl = []; axis_name = '';
for i = 1:numel(axis_candidates)
    nm = axis_candidates{i};
    if isfield(S,nm) && isnumeric(S.(nm)) && isvector(S.(nm))
        wl = S.(nm)(:); axis_name = nm; break;
    end
end
if isempty(wl), error('未找到波长轴变量。'); end
M = numel(wl);

spec_info = struct('name',{},'transposed',{});
for i = 1:numel(fn)
    nm = fn{i}; if strcmpi(nm,axis_name), continue; end
    val = S.(nm);
    if isnumeric(val) && ndims(val)==2
        [h,w] = size(val);
        if h==M && w>=1, spec_info(end+1)=struct('name',nm,'transposed',false); %#ok<SAGROW>
        elseif w==M && h>=1, spec_info(end+1)=struct('name',nm,'transposed',true);  %#ok<SAGROW>
        end
    end
end
if isempty(spec_info), error('MAT 中未发现与波长匹配的样品矩阵'); end

% 按尾号排序，并截取前 FORCE_N 个
nums = nan(numel(spec_info),1);
for i=1:numel(spec_info)
    t = regexp(spec_info(i).name,'(\d+)$','tokens','once');
    if ~isempty(t), nums(i)=str2double(t{1}); end
end
[~,ord] = sortrows([isnan(nums) nums],[1 2]); spec_info = spec_info(ord);
if numel(spec_info) < FORCE_N
    error('仅识别到 %d 个样品，少于 FORCE_N=%d。', numel(spec_info), FORCE_N);
end
spec_info = spec_info(1:FORCE_N);
spec_names = string({spec_info.name}); N = numel(spec_names);
fprintf('[Info] 使用样品：\n'); disp(spec_names.');

%% ===== 读取 Excel，并与样品对齐为 N×Q 的浓度矩阵 =====
if isempty(excel_sheet), T = readtable(excel_xlsx);
else,                    T = readtable(excel_xlsx,'Sheet',excel_sheet);
end
varsL = lower(string(T.Properties.VariableNames));
key_name_candidates = lower(["sample","name","spec","sample_name"]);
key_idx_candidates  = lower(["idx","id","sample_idx","order"]);
key_name_col = find(ismember(varsL,key_name_candidates),1);
key_idx_col  = find(ismember(varsL,key_idx_candidates),1);

Q = numel(response_names);
col_idx = nan(1,Q);
for q=1:Q
    keys = synonyms(response_names{q});
    hitc = find(ismember(varsL, string(keys)), 1, 'first');
    if ~isempty(hitc), col_idx(q)=hitc; end
    if isnan(col_idx(q)), error('Excel 缺少指标列：%s', response_names{q}); end
end

conc_all = nan(N,Q);
if ~isempty(key_name_col) || ~isempty(key_idx_col)
    if ~isempty(key_name_col)
        keyvals = string(T{:,key_name_col}); keyvals = strtrim(lower(keyvals));
        specL = strtrim(lower(spec_names));
        for i=1:N
            hit = find(strcmp(specL(i), keyvals), 1, 'first');
            if isempty(hit), error('Excel 未找到样品名：%s', spec_names(i)); end
            conc_all(i,:) = T{hit, col_idx};
        end
    else
        idxvals = T{:,key_idx_col}; if ~isnumeric(idxvals), error('索引列必须为数值'); end
        for i=1:N
            hit = find(idxvals==i, 1, 'first');
            if isempty(hit), error('Excel 未找到样品索引 = %d', i); end
            conc_all(i,:) = T{hit, col_idx};
        end
    end
else
    if height(T) < N, error('Excel 行数不足 %d 行。', N); end
    conc_all = T{1:N, col_idx};  % 按 1..N 行顺序对齐
end
if any(isnan(conc_all(:))), warning('conc_all 中有 NaN，请检查 Excel。'); end

%% ===== ROI（全谱 or 谱线窗口） =====
use_lines_only = false;
lines_nm = [394.4, 396.15, 279.55, 280.27, 285.21, 373.49, 373.74];
win_nm   = 0.30;
if use_lines_only
    mask = false(size(wl));
    for L = lines_nm, mask = mask | (abs(wl - L) <= win_nm); end
    if ~any(mask), error('谱线窗口掩码为空'); end
else
    mask = true(size(wl));
end
wl_roi = wl(mask); P = nnz(mask);

%% ===== 展开复测为观测级 =====
X_rep = []; Y_rep = []; G_rep = []; R_idx = [];
for si = 1:N
    nm = spec_names(si); Mtx = S.(nm);
    if spec_info(si).transposed, Mtx = Mtx.'; end
    if size(Mtx,1) ~= numel(wl), error('%s 行数=%d 应为 %d', nm, size(Mtx,1), numel(wl)); end
    K = size(Mtx,2);
    for k = 1:K
        vi = Mtx(:,k);
        X_rep(end+1,:) = vi(mask).';           %#ok<SAGROW>
        Y_rep(end+1,:) = conc_all(si,:);       %#ok<SAGROW>
        G_rep(end+1,1) = si;                   %#ok<SAGROW>
        R_idx(end+1,1) = k;                    %#ok<SAGROW>
    end
end
n_obs = size(X_rep,1); unique_samp = unique(G_rep,'stable'); S_used = numel(unique_samp);
fprintf('[Info] 观测=%d | 样品=%d | 特征=%d | 指标=%d\n', n_obs, S_used, P, Q);

%% ===== 选择外测样品集合 =====
if ~isempty(holdout_list)
    test_samples = intersect(unique_samp(:).', holdout_list(:).');
elseif ~isempty(holdout_ratio)
    k = max(1, min(S_used-2, round(S_used*holdout_ratio)));
    test_samples = randsample(S_used, k);
else
    k = max(1, min(S_used-2, holdout_k));
    test_samples = randsample(S_used, k);
end
train_samples = setdiff(unique_samp, test_samples);
if numel(train_samples) < 2, error('训练样品太少：%d', numel(train_samples)); end
fprintf('[Hold-out] 测试样品 idx=%s | 训练样品数=%d\n', mat2str(test_samples), numel(train_samples));

is_tr = ismember(G_rep, train_samples);
is_te = ismember(G_rep, test_samples);

%% ===== 仅用训练样品做“样品分组CV”选择 LV（PLS2） =====
zscore_X = true; zscore_Y = true;
Kfold_grp = min(6, numel(train_samples));
maxLV = min([15, sum(is_tr)-1, P]); rng(rng_seed);

% 为训练样品分折
perm = train_samples(randperm(numel(train_samples)));
fold_of_sample = zeros(numel(train_samples),1);
for i=1:numel(train_samples), fold_of_sample(i) = mod(i-1,Kfold_grp)+1; end
map = containers.Map(num2cell(perm), num2cell(fold_of_sample));
F_rep = nan(n_obs,1);
for i=1:n_obs, if is_tr(i), F_rep(i) = map(G_rep(i)); end, end

% 扫描 LV
MSEz_mean = nan(maxLV,1);
for lv = 1:maxLV
    SSEz = zeros(1,Q); CNT = zeros(1,Q);
    for f = 1:Kfold_grp
        tr = is_tr & (F_rep~=f); te = is_tr & (F_rep==f);
        XT = X_rep(tr,:); XE = X_rep(te,:);
        YT = Y_rep(tr,:); YE = Y_rep(te,:);
        if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
        if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; YTz=(YT-muY)./sigY; YE_z=(YE-muY)./sigY; else, muY=zeros(1,Q); sigY=ones(1,Q); YTz=YT; YE_z=YE; end
        [~,~,~,~,beta] = plsregress(XTz, YTz, lv);
        Yhat_z = [ones(sum(te),1) XEz] * beta;
        Rz = Yhat_z - YE_z; SSEz = SSEz + sum(Rz.^2,1); CNT = CNT + sum(~isnan(Rz),1);
    end
    MSEz_mean(lv) = mean(SSEz ./ max(CNT,1));
end
[best_mse, bestLV] = min(MSEz_mean);
fprintf('[CV-group|train] bestLV=%d | mean MSE_z=%.6g | folds=%d\n', bestLV, best_mse, Kfold_grp);

% 复算训练折外预测（原单位）
yhat_cv = nan(n_obs, Q);
for f = 1:Kfold_grp
    tr = is_tr & (F_rep~=f); te = is_tr & (F_rep==f);
    XT = X_rep(tr,:); XE = X_rep(te,:); YT = Y_rep(tr,:);
    if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
    if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; else, muY=zeros(1,Q); sigY=ones(1,Q); end
    [~,~,~,~,beta] = plsregress(XTz, (YT - muY)./sigY, bestLV);
    Yhat_z = [ones(sum(te),1) XEz] * beta; yhat_cv(te,:) = Yhat_z.*sigY + muY;
end

%% ===== 用训练集重训最终模型；预测外测复测级 =====
XT_all = X_rep(is_tr,:); YT_all = Y_rep(is_tr,:);
if zscore_X, muX_all=mean(XT_all,1); sigX_all=std(XT_all,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(XT_all-muX_all)./sigX_all; else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=XT_all; end
if zscore_Y, muY_all=mean(YT_all,1); sigY_all=std(YT_all,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(YT_all-muY_all)./sigY_all; else, muY_all=zeros(1,Q); sigY_all=ones(1,Q); Yz_all=YT_all; end
[XL_all, YL_all, XS_all, YS_all, beta_all, PCTVAR_all, MSE_all, stats_final] = ...
    plsregress(Xz_all, Yz_all, bestLV); %#ok<ASGLU>

XE_te = X_rep(is_te,:); Kte = sum(is_te);
Xz_te = (XE_te - muX_all)./sigX_all;
Yhat_test_rep = [ones(Kte,1) Xz_te]*beta_all.*sigY_all + muY_all;

%% ===== VIP（总体 + 逐指标） & Hotelling T^2 / Q（训练 vs 测试） =====
A      = bestLV;                       % 最终 LV
pfeat  = size(stats_final.W,1);        % 光谱特征数
W      = stats_final.W(:,1:A);         % p×A, X-weights
Pload  = XL_all(:,1:A);                % p×A, X-loadings
Ttr    = XS_all(:,1:A);                % n_train×A, 训练池得分（标准化域）

% --- 兼容不同 MATLAB 版本的 Y-loadings 形状（A×Q 或 Q×A）---
if size(YL_all,1)==A && size(YL_all,2)==Q
    Qload = YL_all;                    % A×Q
elseif size(YL_all,2)==A && size(YL_all,1)==Q
    Qload = YL_all.';                  % Q×A → A×Q
else
    error('Unexpected size of YL_all: %dx%d (expect A×Q or Q×A).', size(YL_all,1), size(YL_all,2));
end

% --- 总体 VIP（PLS2 聚合）---
SSYcomp = sum(Qload.^2,2)'.*sum(Ttr.^2,1);   % 1×A
normW2  = sum(W.^2,1);                       % 1×A
VIP_all = zeros(pfeat,1);
for j = 1:pfeat
    VIP_all(j) = sqrt( pfeat * sum( SSYcomp .* ((W(j,:).^2)./max(normW2,eps)) ) / max(sum(SSYcomp),eps) );
end
writetable(table(wl_roi, VIP_all, 'VariableNames',{'wavelength_nm','VIP'}), ...
    fullfile(out_dir,'vip_fullspec_final.csv'));
fig_vip = figure('Name','VIP_full_spectrum','Position',[80 80 820 360]);
plot(wl_roi, VIP_all, '-'); hold on; yline(1,'r--','VIP=1','LabelHorizontalAlignment','left'); grid on
xlabel('Wavelength (nm)'); ylabel('VIP'); title(sprintf('VIP（LV=%d, 聚合）', A));
saveas(fig_vip, fullfile(out_dir,'vip_curve.png'));

% --- 逐指标 VIP ---
VIP_by_target = zeros(pfeat, Q);
for qidx = 1:Q
    SSYcomp_q = (Qload(:,qidx)'.^2).*sum(Ttr.^2,1);   % 1×A，仅该目标
    VIP_q = zeros(pfeat,1);
    for j = 1:pfeat
        VIP_q(j) = sqrt( pfeat * sum( SSYcomp_q .* ((W(j,:).^2)./max(normW2,eps)) ) / max(sum(SSYcomp_q),eps) );
    end
    VIP_by_target(:,qidx) = VIP_q;
    % CSV + 曲线
    writetable(table(wl_roi, VIP_q, 'VariableNames',{'wavelength_nm','VIP'}), ...
        fullfile(out_dir, sprintf('vip_fullspec_final_%s.csv', response_names{qidx})));
    fig_vip_q = figure('Name', ['VIP_',response_names{qidx}], 'Position',[100 120 820 360]);
    plot(wl_roi, VIP_q, '-'); hold on; yline(1,'r--','VIP=1','LabelHorizontalAlignment','left'); grid on
    xlabel('Wavelength (nm)'); ylabel('VIP'); title(sprintf('VIP — %s（LV=%d）', response_names{qidx}, A));
    saveas(fig_vip_q, fullfile(out_dir, sprintf('vip_curve_%s.png', response_names{qidx})));
end
colnames_vip = matlab.lang.makeValidName("VIP_"+string(response_names));
vip_all_tbl = [ table(wl_roi, 'VariableNames', {'wavelength_nm'}), array2table(VIP_by_target, 'VariableNames', colnames_vip) ];
writetable(vip_all_tbl, fullfile(out_dir,'vip_fullspec_final_allTargets_wide.csv'));

% --- Hotelling T^2 & Q 残差：训练 vs 测试（标准化域）---
Rproj   = W / (Pload' * W + eps*eye(A));  % p×A
T_train = Xz_all * Rproj;                 % 训练池得分
T_test  = Xz_te  * Rproj;                 % 外测得分
% T²（按训练得分方差标准化）
varTa = var(T_train, 0, 1); varTa(varTa==0)=eps;
T2_tr = sum( (T_train.^2) ./ (ones(size(T_train,1),1)*varTa), 2 );
T2_te = sum( (T_test.^2)  ./ (ones(size(T_test,1),1) *varTa),  2 );
% Q 残差（SPE）
Xhat_tr = T_train * Pload'; E_tr = Xz_all - Xhat_tr; Q_tr = sum(E_tr.^2,2);
Xhat_te = T_test  * Pload'; E_te = Xz_te  - Xhat_te; Q_te = sum(E_te.^2,2);
% 95% 控制限
if exist('prctile','file')
    T2_thr = prctile(T2_tr,95); Q_thr = prctile(Q_tr,95);
else
    T2_thr = quantile(T2_tr,0.95); Q_thr = quantile(Q_tr,0.95);
end
% 图
fig_t2q = figure('Name','T2_vs_Q_train_vs_test','Position',[120 120 720 540]);
loglog(T2_tr, Q_tr, 'o', 'MarkerSize',5, 'Color',[0 0.447 0.741], 'MarkerFaceColor','none'); hold on;
loglog(T2_te, Q_te, 's', 'MarkerSize',6, 'Color',[0.85 0.1 0.1], 'MarkerFaceColor','none');
xline(T2_thr,'k--','T^2 95%','LabelOrientation','horizontal','LabelVerticalAlignment','bottom');
yline(Q_thr, 'k--','Q 95%','LabelHorizontalAlignment','left');
grid on; xlabel('Hotelling T^2 (log)'); ylabel('Q residual / SPE (log)');
title(sprintf('T^2 - Q（train vs test）| LV=%d', A));
legend({'train','test'}, 'Location','best'); hold off;
saveas(fig_t2q, fullfile(out_dir,'t2_q_scatter_train_vs_test.png'));
% 明细导出
T2Q_train = table(find(is_tr), T2_tr, Q_tr, 'VariableNames',{'row_idx_in_train','T2','Q'});
T2Q_test  = table(find(is_te), T2_te, Q_te, 'VariableNames',{'row_idx_in_test','T2','Q'});
writetable(T2Q_train, fullfile(out_dir,'t2q_train.csv'));
writetable(T2Q_test,  fullfile(out_dir,'t2q_test.csv'));

%% ===== 样品级聚合（训练折外 & 外测）并计算指标 =====
train_samples_sorted = sort(train_samples);
test_samples_sorted  = sort(test_samples);
n_tr_samp = numel(train_samples_sorted);
n_te_samp = numel(test_samples_sorted);

% 训练聚合
sample_true_tr = zeros(n_tr_samp,Q); sample_pred_tr_mean=zeros(n_tr_samp,Q);
sample_pred_tr_std=zeros(n_tr_samp,Q); sample_nrep_tr=zeros(n_tr_samp,1);
for ii=1:n_tr_samp
    t = train_samples_sorted(ii);
    idx = (G_rep==t);
    sample_true_tr(ii,:)      = Y_rep(find(idx,1,'first'),:);
    sample_pred_tr_mean(ii,:) = mean(yhat_cv(idx,:),1);
    sample_pred_tr_std(ii,:)  = std(yhat_cv(idx,:),0,1);
    sample_nrep_tr(ii)        = sum(idx);
end

% 测试聚合
sample_true_te = zeros(n_te_samp,Q); sample_pred_te_mean=zeros(n_te_samp,Q);
sample_pred_te_std=zeros(n_te_samp,Q); sample_nrep_te=zeros(n_te_samp,1);
test_rows = find(is_te);
for ii=1:n_te_samp
    t = test_samples_sorted(ii);
    idx = (G_rep(test_rows)==t);
    rows = test_rows(idx);
    sample_true_te(ii,:)      = Y_rep(rows(1),:);
    sample_pred_te_mean(ii,:) = mean(Yhat_test_rep(idx,:),1);
    sample_pred_te_std(ii,:)  = std(Yhat_test_rep(idx,:),0,1);
    sample_nrep_te(ii)        = sum(idx);
end

% 指标
R2_sample_tr=zeros(1,Q); RMSE_sample_tr=zeros(1,Q);
R2_rep_tr=zeros(1,Q);    RMSE_rep_tr=zeros(1,Q);
R2_sample_te=zeros(1,Q); RMSE_sample_te=zeros(1,Q);
R2_rep_te=zeros(1,Q);    RMSE_rep_te=zeros(1,Q);
for q=1:Q
    % 训练样品级
    y=sample_true_tr(:,q); yh=sample_pred_tr_mean(:,q);
    SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
    R2_sample_tr(q)=1-SSE/SST; RMSE_sample_tr(q)=sqrt(mean((y-yh).^2));
    % 训练复测级
    y_rep = Y_rep(is_tr,q); yh_rep = yhat_cv(is_tr,q);
    SSEr=sum((y_rep-yh_rep).^2); SStr=sum((y_rep-mean(y_rep)).^2);
    R2_rep_tr(q)=1-SSEr/SStr; RMSE_rep_tr(q)=sqrt(mean((y_rep-yh_rep).^2));
    % 外测样品级
    if n_te_samp>=2
        y  = sample_true_te(:,q); yh = sample_pred_te_mean(:,q);
        SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
        R2_sample_te(q)=1-SSE/SST; RMSE_sample_te(q)=sqrt(mean((y-yh).^2));
    else
        R2_sample_te(q)=NaN; RMSE_sample_te(q)=NaN;
    end
    % 外测复测级
    y_rep = Y_rep(is_te,q); yh_rep = Yhat_test_rep(:,q);
    SSEr=sum((y_rep-yh_rep).^2); SSt=sum((y_rep-mean(y_rep)).^2);
    R2_rep_te(q)=1-SSEr/SSt; RMSE_rep_te(q)=sqrt(mean((y_rep-yh_rep).^2));
end

%% ===== 训练 LV 曲线 =====
fig1=figure('Name','PLSR_groupCV_train','Position',[60 60 560 380]);
plot(1:numel(MSEz_mean), MSEz_mean, '-o'); grid on; xlabel('LV'); ylabel('mean MSE_z');
title(sprintf('分组CV（训练样品=%d）- 多指标', numel(train_samples)));
saveas(fig1, fullfile(out_dir,'plsr_groupCV_train.png'));

%% ===== “像截图那样”的误差棒图（训练折外 + 外测一起画） =====
for q=1:Q
    fig=figure('Name',['Err_train_with_test_',response_names{q}], 'Position',[640 80 600 480]);
    hold on; grid on; box on;

    % 训练：样品级折外（均值±STD）— 蓝色圆点
    hTrain = errorbar(sample_true_tr(:,q), sample_pred_tr_mean(:,q), sample_pred_tr_std(:,q), ...
        'o', 'LineWidth',1.4, 'MarkerSize',6, 'CapSize',6, 'Color',[0 0.447 0.741]);

    % 外测：样品级（均值±STD）— 红色方块
    hTest = [];
    if n_te_samp>=1
        hTest = errorbar(sample_true_te(:,q), sample_pred_te_mean(:,q), sample_pred_te_std(:,q), ...
            's', 'Color',[0.85 0.1 0.1], 'MarkerFaceColor','none', ...
            'MarkerSize',8, 'LineWidth',1.6, 'CapSize',6, 'LineStyle','none');
    end

    % 1:1 参考线（范围自适应，含训练与外测）
    xymin = min([sample_true_tr(:,q); sample_pred_tr_mean(:,q)-sample_pred_tr_std(:,q)]);
    xymax = max([sample_true_tr(:,q); sample_pred_tr_mean(:,q)+sample_pred_tr_std(:,q)]);
    if n_te_samp>=1
        xymin = min(xymin, min(sample_true_te(:,q)));
        xymax = max(xymax, max(sample_pred_te_mean(:,q)+sample_pred_te_std(:,q)));
    end
    pad = 0.02*(xymax-xymin + eps);
    x1 = xymin - pad; x2 = xymax + pad;
    h11 = plot([x1 x2],[x1 x2],'k--');

    % 标题与图例
    ttl = sprintf('训练折外 | %s | R^2=%.4f, RMSE=%.4g', response_names{q}, R2_sample_tr(q), RMSE_sample_tr(q));
    title(ttl);
    xlabel(['真实 ',response_names{q}]); ylabel(['预测 ',response_names{q},'（均值±STD）']);
    if ~isempty(hTest)
        legend([hTrain h11 hTest], {'train mean±STD','1:1 line','test mean±STD (hold-out)'}, 'Location','best');
    else
        legend([hTrain h11], {'train mean±STD','1:1 line'}, 'Location','best');
    end
    set(gca,'Layer','top'); axis([x1 x2 x1 x2]); hold off;
    saveas(fig, fullfile(out_dir, sprintf('errorbar_train_with_test_%s.png', response_names{q})));
end

%% ===== （可选）只看“外测全部点+误差棒”的图（灰色复测散点 + 红色方块±STD） =====
for q = 1:Q
    if n_te_samp < 1, continue; end
    x_true_te = sample_true_te(:,q);
    y_mean_te = sample_pred_te_mean(:,q);
    y_std_te  = sample_pred_te_std(:,q);

    fig = figure('Name',['Err_test_all_',response_names{q}], 'Position',[640 520 620 480]);
    hold on; grid on; box on;

    % 复测级散点（灰色，半透明）
    test_rows = find(is_te);
    hRep = [];
    for ii = 1:n_te_samp
        t   = test_samples_sorted(ii);
        idx = (G_rep(test_rows) == t);
        sc = scatter(repmat(x_true_te(ii), sum(idx), 1), Yhat_test_rep(idx, q), 14, ...
                     [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha',0.45, 'MarkerEdgeAlpha',0.45);
        if isempty(hRep), hRep = sc; end
    end

    % 样品级误差棒（红色方块）
    hErr = errorbar(x_true_te, y_mean_te, y_std_te, 's', 'Color',[0.85 0.1 0.1], ...
        'MarkerFaceColor','none','MarkerSize',8,'LineWidth',1.6,'CapSize',6,'LineStyle','none');

    % 1:1 线
    xymin = min([x_true_te; y_mean_te - y_std_te; Yhat_test_rep(:,q)]);
    xymax = max([x_true_te; y_mean_te + y_std_te; Yhat_test_rep(:,q)]);
    pad   = 0.02*(xymax-xymin + eps);
    x1 = xymin - pad; x2 = xymax + pad;
    h11 = plot([x1 x2],[x1 x2],'k--');

    % 样品编号标注
    for ii = 1:n_te_samp
        text(x_true_te(ii), y_mean_te(ii), sprintf('  #%d', test_samples_sorted(ii)), ...
            'Color',[0.85 0.1 0.1], 'FontSize',9, 'VerticalAlignment','middle');
    end

    xlabel(['真实 ', response_names{q}]);
    ylabel(['预测 ', response_names{q}, '（均值±STD）']);
    ttl = sprintf('外测 | %s | Sample R^2=%s, RMSE=%s', response_names{q}, ...
        num2str(R2_sample_te(q),'%.4f'), num2str(RMSE_sample_te(q),'%.4g'));
    title(ttl);
    legend([hRep, hErr, h11], {'replicates (test)', 'sample mean±STD (test)', '1:1 line'}, 'Location','best');
    set(gca,'Layer','top'); axis([x1 x2 x1 x2]); hold off;

    saveas(fig, fullfile(out_dir, sprintf('errorbar_test_all_%s.png', response_names{q})));
end

%% ===== 导出结果（与之前一致） =====
% 复测级（区分训练折外与测试外测）
T_rep = table(G_rep, R_idx, 'VariableNames',{'sample_idx','rep_idx'});
for q=1:Q
    T_rep.(['true_',response_names{q}]) = Y_rep(:,q);
    colcv = nan(n_obs,1); colcv(is_tr) = yhat_cv(is_tr,q);
    colte = nan(n_obs,1); colte(is_te) = Yhat_test_rep(:,q);
    T_rep.(['predCV_train_',response_names{q}]) = colcv;
    T_rep.(['predTEST_',response_names{q}])     = colte;
end
writetable(T_rep, fullfile(out_dir,'predictions_replicate_level_train_test.csv'));

% 训练集样品级
T_samp_tr = table(train_samples_sorted(:), sample_nrep_tr, 'VariableNames',{'sample_idx','n_reps'});
for q=1:Q
    T_samp_tr.(['true_',response_names{q}])     = sample_true_tr(:,q);
    T_samp_tr.(['predMean_',response_names{q}]) = sample_pred_tr_mean(:,q);
    T_samp_tr.(['predSTD_',response_names{q}])  = sample_pred_tr_std(:,q);
end
writetable(T_samp_tr, fullfile(out_dir,'predictions_sample_level_train.csv'));

% 测试集样品级
T_samp_te = table(test_samples_sorted(:), sample_nrep_te, 'VariableNames',{'sample_idx','n_reps'});
for q=1:Q
    T_samp_te.(['true_',response_names{q}])     = sample_true_te(:,q);
    T_samp_te.(['predMean_',response_names{q}]) = sample_pred_te_mean(:,q);
    T_samp_te.(['predSTD_',response_names{q}])  = sample_pred_te_std(:,q);
end
writetable(T_samp_te, fullfile(out_dir,'predictions_sample_level_test.csv'));

% 指标汇总
T_metrics = table(response_names(:), repmat(bestLV,Q,1), ...
    R2_sample_tr(:), RMSE_sample_tr(:), R2_rep_tr(:), RMSE_rep_tr(:), ...
    R2_sample_te(:), RMSE_sample_te(:), R2_rep_te(:), RMSE_rep_te(:), ...
    'VariableNames',{'target','bestLV_train','R2_sample_train','RMSE_sample_train','R2_rep_train','RMSE_rep_train', ...
                     'R2_sample_test','RMSE_sample_test','R2_rep_test','RMSE_rep_test'});
writetable(T_metrics, fullfile(out_dir,'metrics_train_test.csv'));

% 系数谱（标准化域；仅训练集拟合）
coefZ = beta_all(2:end,:);  % P×Q
colnames = matlab.lang.makeValidName("betaZ_"+string(response_names));
coef_tbl = [ table(wl_roi, 'VariableNames', {'wavelength_nm'}), array2table(coefZ,'VariableNames',colnames) ];
writetable(coef_tbl, fullfile(out_dir,'plsr_coefficients_roi_multi_train.csv'));

% 保存模型（仅训练集拟合）
model=struct();
model.betaZ=beta_all; model.nLV=bestLV; model.zscore_X=true; model.zscore_Y=true;
model.muX=muX_all; model.sigX=sigX_all; model.muY=muY_all; model.sigY=sigY_all;
model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
model.use_lines_only=false; model.lines_nm=lines_nm(:); model.win_nm=win_nm;
model.holdout_test_idx = test_samples_sorted(:)'; model.holdout_train_idx = train_samples_sorted(:)';
model.random_seed = rng_seed;
save(fullfile(out_dir,'plsr_model_trainOnly_18.mat'),'-struct','model','-v7.3');

%% === 统计并导出：训练用了多少种样品 ===
train_samples_sorted = sort(train_samples);
test_samples_sorted  = sort(test_samples);
n_train_samples = numel(train_samples_sorted);
n_test_samples  = numel(test_samples_sorted);
fprintf('\n[STAT] 训练样品种数 = %d | 测试样品种数 = %d\n', n_train_samples, n_test_samples);
disp(table(train_samples_sorted(:), spec_names(train_samples_sorted).', ...
     'VariableNames', {'sample_idx','sample_name'}));
nrep_tr = arrayfun(@(t) sum(G_rep==t & is_tr), train_samples_sorted);
nrep_te = arrayfun(@(t) sum(G_rep==t & is_te), test_samples_sorted);
disp(table(train_samples_sorted(:), spec_names(train_samples_sorted).', nrep_tr(:), ...
    'VariableNames', {'train_idx','train_name','n_reps_train'}));
disp(table(test_samples_sorted(:),  spec_names(test_samples_sorted).',  nrep_te(:),  ...
    'VariableNames', {'test_idx','test_name','n_reps_test'}));
T_split = table( ...
    [train_samples_sorted(:); test_samples_sorted(:)], ...
    [spec_names(train_samples_sorted).'; spec_names(test_samples_sorted).'], ...
    [repmat("train", n_train_samples,1); repmat("test", n_test_samples,1)], ...
    [nrep_tr(:); nrep_te(:)], ...
    'VariableNames', {'sample_idx','sample_name','split','n_reps'});
writetable(T_split, fullfile(out_dir,'train_test_split.csv'));

fprintf('\n[DONE] 训练折外 + 外测留出 + VIP + T^2/Q 完成。输出目录：%s\n', out_dir);
