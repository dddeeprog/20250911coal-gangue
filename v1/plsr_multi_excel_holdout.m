%% PLSR（多指标；Excel 标签；可随机留出多个样品做外测；其余样品训练+分组CV）
% - 从 MAT（含 wavelengths/wavelength/lambda 与 spec1/spec2/...）读取光谱
% - 从 Excel 读取各指标真值（支持同义列名）
% - 训练：仅用训练样品做“样品分组CV”选 LV；导出训练折外（CV）指标
% - 外测：随机/指定留出多个样品，做真正外测预测与指标
% - 绘图：训练误差棒图上叠加外测样品（红色方块）

clear; clc; close all; rng(2025,'twister');

%% ===== 路径与输出 =====
ave_mat     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\110A110B.mat';
excel_xlsx  = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\targets.xlsx';
excel_sheet = '';   % 指定表名或留空自动
out_dir     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\plsr_out_holdout';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 留出策略（3选1，优先顺序：list > random_k > ratio） =====
holdout_list  = [3 4];     % 直接指定外测样品索引（如 [2 5]），留空则按下面策略
holdout_k     = 2;      % 随机留出的样品数（当 list 为空时生效）
holdout_ratio = [];     % 或按比例（如 0.3），留空忽略
rng_seed      = 2025;   % 随机种子（保证可复现）
rng(rng_seed,'twister');

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
if isempty(wl), error('未找到波长轴变量（尝试 wavelengths/wavelength/lambda/wavelenths）'); end
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
if isempty(spec_info), error('MAT 中未发现与波长长度匹配的样品矩阵'); end
% 按尾部数字排序（如 spec1, spec2, ...）
nums = nan(numel(spec_info),1);
for i=1:numel(spec_info)
    t = regexp(spec_info(i).name,'(\d+)$','tokens','once');
    if ~isempty(t), nums(i)=str2double(t{1}); end
end
[~,ord] = sortrows([isnan(nums) nums],[1 2]); spec_info = spec_info(ord);
spec_names = string({spec_info.name}); N = numel(spec_names);
disp('[Info] 识别样品变量顺序：'); disp(spec_names.');

%% ===== 读取 Excel，并与样品对齐为 N×Q 的浓度矩阵 =====
if isempty(excel_sheet), T = readtable(excel_xlsx);
else,                    T = readtable(excel_xlsx,'Sheet',excel_sheet);
end
varsL = lower(string(T.Properties.VariableNames));
key_name_candidates = lower(["sample","name","spec","sample_name"]);
key_idx_candidates  = lower(["idx","id","sample_idx","order"]);
key_name_col = find(ismember(varsL,key_name_candidates),1);
key_idx_col  = find(ismember(varsL,key_idx_candidates),1);
if isempty(key_name_col) && isempty(key_idx_col)
    error('Excel 需要包含样品标识列（名称列：sample/name/spec 或 索引列：idx/id）。');
end

Q = numel(response_names);
col_idx = nan(1,Q);
for q=1:Q
    keys = synonyms(response_names{q});
    for c = 1:numel(varsL)
        if any(strcmpi(varsL(c), string(keys))), col_idx(q) = c; break; end
    end
    if isnan(col_idx(q))
        error('Excel 中未找到指标列：%s（支持同义名：%s）', response_names{q}, strjoin(keys,','));
    end
end

conc_all = nan(N,Q);
if ~isempty(key_name_col)
    keyvals = string(T{:,key_name_col}); keyvals = strtrim(lower(keyvals));
    specL = strtrim(lower(spec_names));
    for i=1:N
        hit = find(strcmp(specL(i), keyvals), 1, 'first');
        if isempty(hit), error('Excel 未找到与样品变量 "%s" 对应的名称。', spec_names(i)); end
        conc_all(i,:) = T{hit, col_idx};
    end
else
    idxvals = T{:,key_idx_col};
    if ~isnumeric(idxvals), error('索引列必须为数值。'); end
    for i=1:N
        hit = find(idxvals==i, 1, 'first');
        if isempty(hit), error('Excel 未找到样品索引 = %d 的行。', i); end
        conc_all(i,:) = T{hit, col_idx};
    end
end
if any(isnan(conc_all(:))), warning('检测到 conc_all 中存在 NaN，请检查 Excel 是否缺数据。'); end

%% ===== ROI（全谱 or 谱线窗口） =====
use_lines_only = false;
lines_nm = [394.4, 396.15, 279.55, 280.27, 285.21, 373.49, 373.74];
win_nm   = 0.30;
if use_lines_only
    mask = false(size(wl));
    for L = lines_nm, mask = mask | (abs(wl - L) <= win_nm); end
    if ~any(mask), error('谱线窗口掩码为空；请检查 lines_nm/win_nm'); end
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
        X_rep(end+1,:) = vi(mask).';     %#ok<SAGROW>
        Y_rep(end+1,:) = conc_all(si,:); %#ok<SAGROW>
        G_rep(end+1,1) = si;             %#ok<SAGROW>
        R_idx(end+1,1) = k;              %#ok<SAGROW>
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
    k = max(1, min(S_used-2, holdout_k));   % 至少留 1，且保留 >=2 个训练样品
    test_samples = randsample(S_used, k);
end
train_samples = setdiff(unique_samp, test_samples);
if numel(train_samples) < 2
    error('训练样品数不足（%d）。请减少外测留出数量。', numel(train_samples));
end
fprintf('[Hold-out] 测试样品 idx：%s | 训练样品数=%d\n', mat2str(test_samples), numel(train_samples));

is_tr = ismember(G_rep, train_samples);
is_te = ismember(G_rep, test_samples);

%% ===== 仅用训练样品做“样品分组CV”选择 LV（PLS2） =====
zscore_X = true; zscore_Y = true;
Kfold_grp = min(6, numel(train_samples));     % 训练样品做 K 折
maxLV = min([15, sum(is_tr)-1, P]); rng(rng_seed);

% 为训练样品分折
perm = train_samples(randperm(numel(train_samples)));
fold_of_sample = zeros(numel(train_samples),1);
for i=1:numel(train_samples), fold_of_sample(i) = mod(i-1,Kfold_grp)+1; end
map = containers.Map(num2cell(perm), num2cell(fold_of_sample));
F_rep = nan(n_obs,1);
for i=1:n_obs
    if is_tr(i), F_rep(i) = map(G_rep(i)); end
end

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

%% ===== 用训练集全部复测重训最终模型；并预测测试样品 =====
XT_all = X_rep(is_tr,:); YT_all = Y_rep(is_tr,:);
if zscore_X, muX_all=mean(XT_all,1); sigX_all=std(XT_all,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(XT_all-muX_all)./sigX_all; else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=XT_all; end
if zscore_Y, muY_all=mean(YT_all,1); sigY_all=std(YT_all,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(YT_all-muY_all)./sigY_all; else, muY_all=zeros(1,Q); sigY_all=ones(1,Q); Yz_all=YT_all; end
[~,~,~,~,beta_all] = plsregress(Xz_all, Yz_all, bestLV);

% —— 外测预测
XE_te = X_rep(is_te,:); Kte = sum(is_te);
Xz_te = (XE_te - muX_all)./sigX_all;
Yhat_test_rep = [ones(Kte,1) Xz_te]*beta_all.*sigY_all + muY_all;   % 复测级预测（测试）

%% ===== 训练样品：样品级聚合（CV 折外）与指标 =====
train_samples_sorted = sort(train_samples);
n_tr_samp = numel(train_samples_sorted);
sample_true_tr = zeros(n_tr_samp,Q);
sample_pred_tr_mean = zeros(n_tr_samp,Q);
sample_pred_tr_std  = zeros(n_tr_samp,Q);
sample_nrep_tr = zeros(n_tr_samp,1);
for ii=1:n_tr_samp
    t = train_samples_sorted(ii);
    idx = (G_rep==t);
    sample_true_tr(ii,:)      = Y_rep(find(idx,1,'first'),:);
    sample_pred_tr_mean(ii,:) = mean(yhat_cv(idx,:),1);
    sample_pred_tr_std(ii,:)  = std(yhat_cv(idx,:),0,1);
    sample_nrep_tr(ii)        = sum(idx);
end
R2_sample_tr=zeros(1,Q); RMSE_sample_tr=zeros(1,Q);
R2_rep_tr=zeros(1,Q);    RMSE_rep_tr=zeros(1,Q);
for q=1:Q
    % 样品级（训练折外）
    y=sample_true_tr(:,q); yh=sample_pred_tr_mean(:,q);
    SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
    R2_sample_tr(q)=1-SSE/SST; RMSE_sample_tr(q)=sqrt(mean((y-yh).^2));
    % 复测级（训练折外）
    y_rep = Y_rep(is_tr,q); yh_rep = yhat_cv(is_tr,q);
    SSEr = sum((y_rep - yh_rep).^2); SStr = sum((y_rep - mean(y_rep)).^2);
    R2_rep_tr(q)=1 - SSEr/SStr; RMSE_rep_tr(q)=sqrt(mean((y_rep - yh_rep).^2));
end

%% ===== 测试样品：聚合与指标（样品级/复测级） =====
test_samples_sorted = sort(test_samples);
n_te_samp = numel(test_samples_sorted);
% 聚合
sample_true_te = zeros(n_te_samp,Q);
sample_pred_te_mean = zeros(n_te_samp,Q);
sample_pred_te_std  = zeros(n_te_samp,Q);
sample_nrep_te = zeros(n_te_samp,1);

% 映射：测试复测的行索引（is_te）对应哪个样品
test_rows = find(is_te);
for ii=1:n_te_samp
    t = test_samples_sorted(ii);
    idx = (G_rep(test_rows)==t);             % 在测试集中的行
    rows = test_rows(idx);
    sample_true_te(ii,:)      = Y_rep(rows(1),:);
    sample_pred_te_mean(ii,:) = mean(Yhat_test_rep(idx,:),1);
    sample_pred_te_std(ii,:)  = std(Yhat_test_rep(idx,:),0,1);
    sample_nrep_te(ii)        = sum(idx);
end

% 样品级与复测级（外测）指标
R2_sample_te=zeros(1,Q); RMSE_sample_te=zeros(1,Q);
R2_rep_te=zeros(1,Q);    RMSE_rep_te=zeros(1,Q);
for q=1:Q
    % 样品级：跨测试样品计算 R²/RMSE
    if n_te_samp >= 2
        y  = sample_true_te(:,q); yh = sample_pred_te_mean(:,q);
        SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
        R2_sample_te(q)=1-SSE/SST; RMSE_sample_te(q)=sqrt(mean((y-yh).^2));
    else
        R2_sample_te(q)=NaN; RMSE_sample_te(q)=NaN;
    end
    % 复测级：跨所有测试复测计算
    y_rep = Y_rep(is_te,q); yh_rep = Yhat_test_rep(:,q);
    SSEr = sum((y_rep - yh_rep).^2); SSt = sum((y_rep - mean(y_rep)).^2);
    R2_rep_te(q)= 1 - SSEr/SSt; RMSE_rep_te(q)=sqrt(mean((y_rep - yh_rep).^2));
end

%% ===== 打印测试摘要 =====
fprintf('\n[TEST] 外测样品 idx=%s | 数量=%d\n', mat2str(test_samples_sorted), n_te_samp);
for q=1:Q
    fprintf('  %-9s | Sample-level: R2=%.4f RMSE=%.4g | Replicate-level: R2=%.4f RMSE=%.4g\n',...
        response_names{q}, R2_sample_te(q), RMSE_sample_te(q), R2_rep_te(q), RMSE_rep_te(q));
end

%% ===== 可视化（训练 LV 曲线 + 训练样品误差棒，并叠加外测样品为红色方块） =====
% 训练 LV 曲线
fig1=figure('Name','PLSR_groupCV_train','Position',[80 80 560 380]);
plot(1:numel(MSEz_mean), MSEz_mean, '-o'); grid on; xlabel('LV'); ylabel('mean MSE_z');
title(sprintf('分组CV（训练样品=%d）- 多指标', numel(train_samples)));
saveas(fig1, fullfile(out_dir,'plsr_groupCV_train.png'));

% 每指标误差棒 + 外测叠加
for q=1:Q
    fig=figure('Name',['Err_train_',response_names{q}],'Position',[660 80 560 440]);
    hold on; grid on; box on;
    % 训练：样品级折外预测（均值±STD）
    hTrain = errorbar(sample_true_tr(:,q), sample_pred_tr_mean(:,q), sample_pred_tr_std(:,q), ...
        'o','LineWidth',1.4,'MarkerSize',6,'CapSize',6,'Color',[0 0.447 0.741]);
    % 1:1 参考线
    xymin=min([sample_true_tr(:,q); sample_pred_tr_mean(:,q)-sample_pred_tr_std(:,q)]);
    xymax=max([sample_true_tr(:,q); sample_pred_tr_mean(:,q)+sample_pred_tr_std(:,q)]);
    if n_te_samp>=1
        xymin=min(xymin, min(sample_true_te(:,q)));
        xymax=max(xymax, max(sample_pred_te_mean(:,q)+sample_pred_te_std(:,q)));
    end
    h11 = plot([xymin xymax],[xymin xymax],'k--');
    % 外测：样品级预测均值（红色方块）
    if n_te_samp>=1
        hTest = plot(sample_true_te(:,q), sample_pred_te_mean(:,q), 'rs', 'MarkerSize',8, 'LineWidth',1.5);
        legend([hTrain h11 hTest], {'train mean±STD','1:1 line','test mean (hold-out)'}, 'Location','best');
    else
        legend([hTrain h11], {'train mean±STD','1:1 line'}, 'Location','best');
    end
    xlabel(['真实 ',response_names{q}]); ylabel(['预测 ',response_names{q},'（均值±STD）']);
    title(sprintf('训练折外 | %s | R^2=%.4f, RMSE=%.4g, LV=%d', response_names{q}, R2_sample_tr(q), RMSE_sample_tr(q), bestLV));
    set(gca,'Layer','top');
    hold off;
    saveas(fig, fullfile(out_dir, sprintf('errorbar_train_with_test_%s.png', response_names{q})));
end

%% ===== 导出 CSV =====
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

% 训练/测试指标汇总
T_metrics = table(response_names(:), repmat(bestLV,Q,1), ...
    R2_sample_tr(:), RMSE_sample_tr(:), R2_rep_tr(:), RMSE_rep_tr(:), ...
    R2_sample_te(:), RMSE_sample_te(:), R2_rep_te(:), RMSE_rep_te(:), ...
    'VariableNames',{'target','bestLV_train','R2_sample_train','RMSE_sample_train','R2_rep_train','RMSE_rep_train', ...
                     'R2_sample_test','RMSE_sample_test','R2_rep_test','RMSE_rep_test'});
writetable(T_metrics, fullfile(out_dir,'metrics_train_test.csv'));

% 系数谱（标准化域；仅训练集拟合的模型）
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
save(fullfile(out_dir,'plsr_model_trainOnly.mat'),'-struct','model','-v7.3');

fprintf('\n[DONE] 训练/外测流程完成（含叠加外测样品的训练图）。输出目录：%s\n', out_dir);
