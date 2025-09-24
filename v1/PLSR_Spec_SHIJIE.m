%% PLSR_Spec_SHIJIE  —  多指标 PLSR（不平均，所有复测参与建模；修复 agg_by_sample 越界）
% - 路径：选文件夹 → 自动在该文件夹及子目录搜索 ave.mat 和 浓度Excel
% - 数据：从 ave.mat 自动识别波长轴 + 样品矩阵（MxK 或 KxM）
% - 组分：TotalS/Ash/Volatile/HHV/C/H/N（可改），从 Excel 名称列或索引列对齐
% - 特征：全谱 或 多条谱线 ±win_nm 窗口拼接（可自动读“特征谱线.xlsx”）
% - 评估：'holdout'（按样品划分训练/测试；训练侧做加权线性再校正）或 'cv'（按样品分组CV）
% - 修复点：在聚合前将 yhat_tr / yhat_te 扩展为与 Y_rep 同长度的 full 矩阵，避免逻辑索引越界

clear; clc; close all; rng(2025,'twister');

%% ===== 路径与输出目录 =====
open_output_dir = true;
root_dir = uigetdir(pwd, '请选择包含 ave.mat 与 浓度Excel 的文件夹（或其上层）');
if isequal(root_dir,0), error('已取消。'); end
cd(root_dir);

% 找 MAT
mat_list = [dir(fullfile(root_dir,'*.mat')); dir(fullfile(root_dir,'**','*.mat'))];
if isempty(mat_list)
    [mfile,mpath] = uigetfile({'*.mat','MAT 文件 (*.mat)'}, '未找到 MAT，请手动选择');
    if isequal(mfile,0), error('未选择 MAT'); end
    ave_mat = fullfile(mpath, mfile);
else
    names = string({mat_list.name});
    prio  = find(contains(lower(names),'ave'),1); if isempty(prio), prio=1; end
    ave_mat = fullfile(mat_list(prio).folder, mat_list(prio).name);
end
fprintf('[Info] 使用 MAT: %s\n', ave_mat);

% 找 浓度 Excel（多指标）
xls_list = [dir(fullfile(root_dir,'*.xlsx')); dir(fullfile(root_dir,'*.xls')); ...
            dir(fullfile(root_dir,'**','*.xlsx')); dir(fullfile(root_dir,'**','*.xls'))];
% 优先含 target/conc/content/浓度/成分 关键词
pick = find(contains(lower(string({xls_list.name})), {'target','conc','content','浓度','成分'}), 1);
if isempty(pick)
    [xfile,xpath] = uigetfile({'*.xlsx;*.xls','Excel 文件'}, '选择浓度Excel');
    if isequal(xfile,0), error('未选择 Excel'); end
    excel_xlsx = fullfile(xpath, xfile);
else
    excel_xlsx = fullfile(xls_list(pick).folder, xls_list(pick).name);
end
fprintf('[Info] 使用浓度 Excel: %s\n', excel_xlsx);

% 找“特征谱线.xlsx”（可选）
lines_xlsx = '';
cand = [dir(fullfile(root_dir,'*特征*谱线*.xlsx')); dir(fullfile(root_dir,'**','*特征*谱线*.xlsx'))];
if ~isempty(cand), lines_xlsx = fullfile(cand(1).folder, cand(1).name); end

out_dir = fullfile(fileparts(ave_mat), 'plsr_out_multi_xlsx');
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 目标组分与同义名 =====
response_names = {'TotalS','Ash','Volatile','HHV','C','H','N'};  % 可改
Q = numel(response_names);
synonyms = containers.Map();
synonyms('TotalS')  = lower({'TotalS','S','Sulfur','Total_S','S_total'});
synonyms('Ash')     = lower({'Ash','AshContent','Ash_Content'});
synonyms('Volatile')= lower({'Volatile','VM','Vol','VolatileMatter'});
synonyms('HHV')     = lower({'HHV','GCV','CalorificValue','HighHeatingValue'});
synonyms('C')       = lower({'C','Carbon'});
synonyms('H')       = lower({'H','Hydrogen'});
synonyms('N')       = lower({'N','Nitrogen'});
require_N_equals_6  = true;   % 仅提醒

%% ===== 模式与超参 =====
eval_mode   = 'holdout';   % 'holdout' 或 'cv'
train_idx   = [];          % 指定样品编号（如 [1 2 3 4]）；留空则按 train_n
train_n     = 4;           % holdout 训练样品数
random_train= false;       % true 随机选 n 个；false 取前 n 个

lv_select   = 'auto';      % 'auto' | 'fixed'
lv_fixed    = 3;           % 固定 LV 时使用
varY_thresh = 0.95;        % auto：累计解释Y方差阈值
zscore_X    = true;        % X 标准化（训练参数）
zscore_Y    = true;        % Y 标准化（训练参数）

% CV 参数
Kfold_grp   = 6;
maxLV_cv    = 15;
random_seed = 2025;

%% ===== 特征：全谱 or 多线窗口 =====
use_lines_only       = false;      % true=仅窗口；false=全谱
win_nm               = 0.30;      % 单线半宽（±win_nm）
lines_from_xlsx_first= true;      % 若能读到“特征谱线.xlsx”则优先用其第一列
lines_nm_default     = [394.4, 396.15, 279.55, 280.27, 285.21, 373.49, 373.74];

%% ===== 读取 MAT（波长轴+样品矩阵）=====
S = load(ave_mat); fn = fieldnames(S);
axis_candidates = {'wavelengths','wavelength','lambda','wavelenths'};
wl = []; axis_name = '';
for i=1:numel(axis_candidates)
    nm = axis_candidates{i};
    if isfield(S,nm) && isnumeric(S.(nm)) && isvector(S.(nm))
        wl = S.(nm)(:); axis_name = nm; break;
    end
end
if isempty(wl), error('MAT 中未找到波长轴变量（wavelengths / wavelength / lambda / wavelenths）'); end
M = numel(wl);

% 收集样品矩阵（MxK 或 KxM）
spec_info = struct('name',{},'transposed',{});
for i=1:numel(fn)
    nm = fn{i}; if strcmpi(nm,axis_name), continue; end
    A = S.(nm);
    if isnumeric(A) && ismatrix(A)
        [h,w] = size(A);
        if h==M && w>=1, spec_info(end+1)=struct('name',nm,'transposed',false); %#ok<SAGROW>
        elseif w==M && h>=1, spec_info(end+1)=struct('name',nm,'transposed',true); %#ok<SAGROW>
        end
    end
end
if isempty(spec_info), error('MAT 中未发现与波长长度匹配的样品矩阵'); end
% 名字末尾数字排序
nums = nan(numel(spec_info),1);
for i=1:numel(spec_info)
    t = regexp(spec_info(i).name,'(\d+)$','tokens','once'); if ~isempty(t), nums(i)=str2double(t{1}); end
end
[~,ord] = sortrows([isnan(nums) nums],[1 2]); spec_info = spec_info(ord);
spec_names = string({spec_info.name}); N = numel(spec_names);
disp('[Info] 样品变量顺序：'); disp(spec_names.');
if require_N_equals_6 && N~=6
    warning('当前样品数 N=%d（期望=6）；继续按全部样品处理。', N);
end

%% ===== 读取浓度 Excel，并对齐到 N×Q =====
T = readtable(excel_xlsx); varsL = lower(string(T.Properties.VariableNames));
key_name_candidates = lower(["sample","name","spec","sample_name"]);
key_idx_candidates  = lower(["idx","id","sample_idx","order"]);
key_name_col = find(ismember(varsL, key_name_candidates),1);
key_idx_col  = find(ismember(varsL, key_idx_candidates),1);
if isempty(key_name_col) && isempty(key_idx_col)
    warning('Excel 未检测到样品标识列（name/idx），将按行号 1..N 对齐。');
end

% 每个指标找列
col_idx = nan(1,Q);
for q=1:Q
    keys = synonyms(response_names{q});
    hit = find(ismember(varsL, string(keys)), 1);
    if isempty(hit), error('Excel 未找到指标列：%s', response_names{q}); end
    col_idx(q) = hit;
end

conc_all = nan(N,Q);
if ~isempty(key_name_col)
    keyvals = string(T{:,key_name_col}); keyvals = strtrim(lower(keyvals));
    specL   = strtrim(lower(spec_names));
    for i=1:N
        r = find(strcmp(specL(i), keyvals), 1);
        if isempty(r), error('Excel 未找到样品名：%s', spec_names(i)); end
        conc_all(i,:) = T{r, col_idx};
    end
elseif ~isempty(key_idx_col)
    idxvals = T{:,key_idx_col};
    for i=1:N
        r = find(idxvals==i, 1);
        if isempty(r), error('Excel 未找到样品 idx=%d', i); end
        conc_all(i,:) = T{r, col_idx};
    end
else
    if height(T) < N, error('Excel 行数(%d) < 样品数(%d)', height(T), N); end
    conc_all(:,:) = T{1:N, col_idx};
end
if any(isnan(conc_all(:))), warning('conc_all 存在 NaN，请检查 Excel 数据。'); end

%% ===== 特征掩码（全谱 or 多线窗口）=====
if use_lines_only
    if ~isempty(lines_xlsx) && lines_from_xlsx_first
        try
            TT = readtable(lines_xlsx); Lcand = TT{:,1}; Lcand = Lcand(:); Lcand = Lcand(isfinite(Lcand));
            if ~isempty(Lcand), lines_nm = Lcand(:).'; else, lines_nm = lines_nm_default; end
        catch
            warning('读取特征谱线Excel失败，改用默认 lines_nm'); lines_nm = lines_nm_default;
        end
    else
        lines_nm = lines_nm_default;
    end
    mask = false(size(wl));
    for L = lines_nm
        mask = mask | (abs(wl - L) <= win_nm);
    end
    if ~any(mask), error('谱线窗口掩码为空，请检查 lines_nm / win_nm'); end
else
    mask = true(size(wl));
end
wl_roi = wl(mask); P = nnz(mask);
fprintf('[Info] 特征维 P=%d（%s）\n', P, ternary(use_lines_only,'多线窗口','全谱'));

%% ===== 展开为复测级观测 =====
% X_rep: [n_obs × P];  Y_rep: [n_obs × Q];  G_rep: [n_obs×1] 样品编号； R_idx: 复测序号
X_rep = []; Y_rep = []; G_rep = []; R_idx = [];
for si = 1:N
    nm = spec_names(si); A = S.(nm); if spec_info(si).transposed, A = A.'; end  % MxK
    if size(A,1) ~= M, error('%s 行数=%d，应为 %d', nm, size(A,1), M); end
    K = size(A,2);
    % 取 ROI
    Aroi = A(mask, :).';      % K × P
    X_rep = [X_rep; Aroi];    %#ok<AGROW>
    Y_rep = [Y_rep; repmat(conc_all(si,:), K, 1)]; %#ok<AGROW>
    G_rep = [G_rep; repmat(si, K, 1)]; %#ok<AGROW>
    R_idx = [R_idx; (1:K)'];  %#ok<AGROW>
end
n_obs = size(X_rep,1); fprintf('[Info] 复测级观测数 = %d\n', n_obs);

%% ===== HOLDOUT 或 分组CV =====
if strcmpi(eval_mode,'holdout')
    % —— 样品划分 —— 
    all_samp = 1:N;
    if isempty(train_idx)
        if random_train
            rng(2025); train_idx = sort(randsample(all_samp, min(train_n, N)));
        else
            train_idx = all_samp(1:min(train_n, N));
        end
    else
        train_idx = unique(train_idx(:)'); 
    end
    test_idx = setdiff(all_samp, train_idx, 'stable');
    if numel(train_idx) < 2, error('训练样品数过少（<2）。'); end
    fprintf('[Holdout] Train=%s | Test=%s\n', mat2str(train_idx), mat2str(test_idx));

    % —— 复测级划分 —— 
    tr_obs = ismember(G_rep, train_idx);
    te_obs = ismember(G_rep, test_idx);
    XT = X_rep(tr_obs,:); XE = X_rep(te_obs,:);
    YT = Y_rep(tr_obs,:); YE = Y_rep(te_obs,:);

    % —— 标准化（仅用训练集参数） —— 
    if zscore_X
        muX = mean(XT,1); sigX = std(XT,[],1); sigX(sigX==0)=eps;
        XTz = (XT - muX)./sigX;  XEz = (XE - muX)./sigX;
    else
        muX = zeros(1,P); sigX = ones(1,P); XTz = XT; XEz = XE;
    end
    if zscore_Y
        muY = mean(YT,1); sigY = std(YT,[],1); sigY(sigY==0)=eps;
        YTz = (YT - muY)./sigY;
    else
        muY = zeros(1,Q); sigY = ones(1,Q); YTz = YT;
    end

    % —— 选择 LV（训练集） —— 
    maxLV_eff = max(1, min([size(XTz,1)-1, size(XTz,2), 25]));
    if strcmpi(lv_select,'fixed')
        bestLV = max(1, min(lv_fixed, maxLV_eff));
    else
        [~,~,~,~,~,PCTVAR] = plsregress(XTz, YTz, maxLV_eff);
        cumY = cumsum(PCTVAR(2,1:maxLV_eff));
        ii = find(cumY >= varY_thresh, 1, 'first'); if isempty(ii), ii = maxLV_eff; end
        bestLV = ii;
    end
    fprintf('[Holdout] 选定 LV=%d\n', bestLV);

    % —— 训练与预测（标准化域→原单位） —— 
    [~,~,~,~,betaZ] = plsregress(XTz, YTz, bestLV);  % (P+1)×Q
    yhat_tr = [ones(sum(tr_obs),1) XTz] * betaZ;     % 标准化域
    yhat_tr = yhat_tr.*sigY + muY;                   % 原单位
    if any(te_obs)
        yhat_te = [ones(sum(te_obs),1) XEz] * betaZ; % 标准化域
        yhat_te = yhat_te.*sigY + muY;               % 原单位
    else
        yhat_te = zeros(0,Q);
    end

    % ★★★ 修复点：扩展为“全长”矩阵，再做样品级聚合（避免逻辑索引越界） ★★★
    yhat_tr_full = nan(n_obs, Q);   % 与 Y_rep 同行数
    yhat_te_full = nan(n_obs, Q);
    yhat_tr_full(tr_obs, :) = yhat_tr;
    yhat_te_full(te_obs, :) = yhat_te;

    % —— 复测级指标（每指标） —— 
    R2_train_rep = nan(1,Q); RMSE_train_rep = nan(1,Q);
    R2_test_rep  = nan(1,Q); RMSE_test_rep  = nan(1,Q);
    for q=1:Q
        yt = YT(:,q); yhT = yhat_tr(:,q);
        R2_train_rep(q)   = 1 - sum((yt - yhT).^2)/sum((yt - mean(yt)).^2);
        RMSE_train_rep(q) = sqrt(mean((yt - yhT).^2));
        if any(te_obs)
            ye = YE(:,q); yhE = yhat_te(:,q);
            R2_test_rep(q)   = 1 - sum((ye - yhE).^2)/sum((ye - mean(ye)).^2);
            RMSE_test_rep(q) = sqrt(mean((ye - yhE).^2));
        end
    end

    % —— 样品级聚合（均值±STD；用 *_full 与全长掩码） —— 
    [train_true_s, train_mean_s, train_std_s] = agg_by_sample(Y_rep, yhat_tr_full, G_rep, train_idx, tr_obs);
    [test_true_s,  test_mean_s,  test_std_s ] = agg_by_sample(Y_rep, yhat_te_full, G_rep, test_idx,  te_obs);

    % —— 训练侧“带权线性再校正”，并评估测试 —— 
    slope = zeros(1,Q); intercept = zeros(1,Q);
    R2_test_s = nan(1,Q); RMSE_test_s = nan(1,Q);
    RSD_train_unCal = nan(1,Q); RSD_test_unCal = nan(1,Q);

    for q=1:Q
        x_tr = train_mean_s(:,q); y_tr = train_true_s(:,q);
        w_tr = 1 ./ (train_std_s(:,q).^2 + eps);
        [slope(q), intercept(q)] = wlinfit(x_tr, y_tr, w_tr);

        % 应用到测试
        x_te = test_mean_s(:,q); y_te = test_true_s(:,q);
        y_te_cal = slope(q).*x_te + intercept(q);

        if ~isempty(y_te)
            SSE = sum((y_te - y_te_cal).^2); SST = sum((y_te - mean(y_te)).^2);
            R2_test_s(q)   = 1 - SSE/SST;
            RMSE_test_s(q) = sqrt(mean((y_te - y_te_cal).^2));
        end

        % RSD（未校正的一致性）——与你原逻辑一致
        RSD_train_unCal(q) = mean(train_std_s(:,q) ./ max(train_mean_s(:,q),eps));
        RSD_test_unCal(q)  = mean(test_std_s(:,q)  ./ max(test_mean_s(:,q), eps));

        % 作图（测试样品级）
        if ~isempty(x_te)
            fig=figure('Name',['Holdout_Errorbar_',response_names{q}], 'Position',[640 80 560 440]);
            errorbar(y_te, y_te_cal, test_std_s(:,q), 'o','LineWidth',1.4,'MarkerSize',6,'CapSize',6);
            hold on; grid on; box on;
            xymin=min([y_te; y_te_cal - test_std_s(:,q)]); xymax=max([y_te; y_te_cal + test_std_s(:,q)]);
            plot([xymin xymax],[xymin xymax],'k--');
            xlabel(['真实 ',response_names{q}]); ylabel(['预测(线性校正) ',response_names{q}]);
            title(sprintf('Holdout Test | %s | R^2=%.4f, RMSE=%.4g, LV=%d', response_names{q}, R2_test_s(q), RMSE_test_s(q), bestLV));
            saveas(fig, fullfile(out_dir, sprintf('holdout_errorbar_sample_%s.png', response_names{q})));
        end
    end

    % —— 导出 —— 
    % 复测级（train/test）
    T_tr = table(G_rep(tr_obs), R_idx(tr_obs), 'VariableNames',{'sample_idx','rep_idx'});
    for q=1:Q, T_tr.(['true_',response_names{q}])=YT(:,q);  T_tr.(['pred_',response_names{q}])=yhat_tr(:,q); end
    writetable(T_tr, fullfile(out_dir,'holdout_train_replicate_multi.csv'));

    if any(te_obs)
        T_te = table(G_rep(te_obs), R_idx(te_obs), 'VariableNames',{'sample_idx','rep_idx'});
        for q=1:Q, T_te.(['true_',response_names{q}])=YE(:,q);  T_te.(['pred_',response_names{q}])=yhat_te(:,q); end
        writetable(T_te, fullfile(out_dir,'holdout_test_replicate_multi.csv'));
    end

    % 样品级（train/test）
    Ts_tr = table(train_idx(:),'VariableNames',{'sample_idx'}); Ts_tr.n_reps = arrayfun(@(sid) sum(tr_obs & (G_rep==sid)), train_idx(:));
    for q=1:Q
        Ts_tr.(['true_',response_names{q}])           = train_true_s(:,q);
        Ts_tr.(['predMean_unCal_',response_names{q}]) = train_mean_s(:,q);
        Ts_tr.(['predSTD_unCal_', response_names{q}]) = train_std_s(:,q);
        Ts_tr.(['predMean_cal_',  response_names{q}]) = slope(q).*train_mean_s(:,q) + intercept(q);
    end
    writetable(Ts_tr, fullfile(out_dir,'holdout_train_sample_multi.csv'));

    if any(te_obs)
        Ts_te = table(test_idx(:),'VariableNames',{'sample_idx'}); Ts_te.n_reps = arrayfun(@(sid) sum(te_obs & (G_rep==sid)), test_idx(:));
        for q=1:Q
            Ts_te.(['true_',response_names{q}])           = test_true_s(:,q);
            Ts_te.(['predMean_unCal_',response_names{q}]) = test_mean_s(:,q);
            Ts_te.(['predSTD_unCal_', response_names{q}]) = test_std_s(:,q);
            Ts_te.(['predMean_cal_',  response_names{q}]) = slope(q).*test_mean_s(:,q) + intercept(q);
        end
        writetable(Ts_te, fullfile(out_dir,'holdout_test_sample_multi.csv'));
    end

    % 指标表
    Tm = table(response_names(:), R2_train_rep(:), RMSE_train_rep(:), R2_test_rep(:), RMSE_test_rep(:), ...
               R2_test_s(:), RMSE_test_s(:), RSD_train_unCal(:), RSD_test_unCal(:), ...
               'VariableNames',{'target','R2_train_rep','RMSE_train_rep','R2_test_rep','RMSE_test_rep', ...
                                'R2_test_sample_cal','RMSE_test_sample_cal','RSD_train_unCal','RSD_test_unCal'});
    writetable(Tm, fullfile(out_dir,'metrics_by_target_holdout.csv'));

    % 系数谱（标准化域）
    coefZ = betaZ(2:end,:); coef_tbl = table(wl_roi, coefZ); 
    coef_tbl.Properties.VariableNames=[{'wavelength_nm'}, strcat('betaZ_',response_names)];
    writetable(coef_tbl, fullfile(out_dir,'plsr_coefficients_roi_multi.csv'));

    % 保存模型
    model=struct();
    model.betaZ=betaZ; model.nLV=bestLV; model.zscore_X=zscore_X; model.zscore_Y=zscore_Y;
    model.muX=muX; model.sigX=sigX; model.muY=muY; model.sigY=sigY;
    model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
    model.train_idx=train_idx(:); model.test_idx=test_idx(:);
    model.use_lines_only=use_lines_only; model.lines_nm=exist('lines_nm','var')*lines_nm(:); model.win_nm=win_nm;
    save(fullfile(out_dir,'plsr_model_multi_xlsx_holdout.mat'),'-struct','model','-v7.3');

else
    %% ===== 分组CV（按样品） =====
    rng(random_seed);
    perm = randperm(N); Keff = min(Kfold_grp, N);
    fold_of_sample = zeros(N,1); for i=1:N, fold_of_sample(perm(i)) = mod(i-1, Keff)+1; end
    F_rep = fold_of_sample(G_rep);   % 复测级折别

    maxLV = min([maxLV_cv, n_obs-1, P]);
    MSEz_mean = nan(maxLV,1);
    for lv = 1:maxLV
        SSEz = zeros(1,Q); CNT = zeros(1,Q);
        for f = 1:Keff
            tr = F_rep~=f; te = ~tr;
            XT = X_rep(tr,:); XE = X_rep(te,:);
            YT = Y_rep(tr,:); YE = Y_rep(te,:);
            % 标准化
            if zscore_X, muX = mean(XT,1); sigX = std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
            if zscore_Y, muY = mean(YT,1); sigY = std(YT,[],1); sigY(sigY==0)=eps; YTz=(YT-muY)./sigY; YE_z=(YE-muY)./sigY; else, muY=zeros(1,Q); sigY=ones(1,Q); YTz=YT; YE_z=YE; end
            [~,~,~,~,beta] = plsregress(XTz, YTz, lv);
            Yhat_z = [ones(sum(te),1) XEz] * beta;
            Rz = Yhat_z - YE_z; SSEz = SSEz + sum(Rz.^2,1); CNT = CNT + sum(~isnan(Rz),1);
        end
        MSEz_mean(lv) = mean(SSEz ./ max(CNT,1));
    end
    [best_mse, bestLV] = min(MSEz_mean); %#ok<ASGLU>
    fprintf('[CV-group] bestLV=%d | mean MSEz=%.6g | folds=%d\n', bestLV, Keff);

    % 复算一次 CV 预测（原单位）
    yhat_cv = nan(n_obs,Q);
    for f = 1:Keff
        tr = F_rep~=f; te = ~tr;
        XT = X_rep(tr,:); XE = X_rep(te,:); YT = Y_rep(tr,:);
        if zscore_X, muX = mean(XT,1); sigX = std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
        if zscore_Y, muY = mean(YT,1); sigY = std(YT,[],1); sigY(sigY==0)=eps; else, muY=zeros(1,Q); sigY=ones(1,Q); end
        [~,~,~,~,beta] = plsregress(XTz, (YT - muY)./sigY, bestLV);
        Yhat_z = [ones(sum(te),1) XEz] * beta; yhat_cv(te,:) = Yhat_z.*sigY + muY;
    end

    % 样品级聚合
    sample_true = zeros(N,Q); sample_pred_mean=zeros(N,Q); sample_pred_std=zeros(N,Q); sample_nrep=zeros(N,1);
    for t=1:N
        ii = (G_rep==t);
        sample_true(t,:)      = Y_rep(find(ii,1,'first'),:);
        sample_pred_mean(t,:) = mean(yhat_cv(ii,:),1);
        sample_pred_std(t,:)  = std( yhat_cv(ii,:),0,1);
        sample_nrep(t)        = sum(ii);
    end
    R2_sample=zeros(1,Q); RMSE_sample=zeros(1,Q);
    for q=1:Q
        y=sample_true(:,q); yh=sample_pred_mean(:,q);
        R2_sample(q)=1 - sum((y-yh).^2)/sum((y-mean(y)).^2);
        RMSE_sample(q)=sqrt(mean((y-yh).^2));
    end

    % 导出/作图
    fig1=figure('Name','PLSR_groupCV_multi','Position',[80 80 560 380]);
    plot(1:numel(MSEz_mean), MSEz_mean, '-o'); grid on; xlabel('LV'); ylabel('mean MSE_z'); title('分组CV（样品）- 多指标');
    saveas(fig1, fullfile(out_dir, 'plsr_groupCV_multi.png'));

    for q=1:Q
        fig=figure('Name',['Err_',response_names{q}],'Position',[660 80 560 440]);
        errorbar(sample_true(:,q), sample_pred_mean(:,q), sample_pred_std(:,q), 'o','LineWidth',1.4,'MarkerSize',6,'CapSize',6);
        hold on; grid on; box on;
        xymin=min([sample_true(:,q); sample_pred_mean(:,q)-sample_pred_std(:,q)]);
        xymax=max([sample_true(:,q); sample_pred_mean(:,q)+sample_pred_std(:,q)]);
        plot([xymin xymax],[xymin xymax],'k--');
        xlabel(['真实 ',response_names{q}]); ylabel(['预测 ',response_names{q},'（均值±STD）']);
        title(sprintf('%s | R^2=%.4f, RMSE=%.4g, LV=%d', response_names{q}, R2_sample(q), RMSE_sample(q), bestLV));
        saveas(fig, fullfile(out_dir, sprintf('errorbar_sample_%s.png', response_names{q})));
    end

    T_rep = table(G_rep, R_idx, 'VariableNames',{'sample_idx','rep_idx'});
    for q=1:Q, T_rep.(['true_',response_names{q}])=Y_rep(:,q); T_rep.(['predCV_',response_names{q}])=yhat_cv(:,q); end
    writetable(T_rep, fullfile(out_dir,'predictions_replicate_level_multi.csv'));
    T_samp = table((1:N).', sample_nrep, 'VariableNames',{'sample_idx','n_reps'});
    for q=1:Q
        T_samp.(['true_',response_names{q}])     = sample_true(:,q);
        T_samp.(['predMean_',response_names{q}]) = sample_pred_mean(:,q);
        T_samp.(['predSTD_',response_names{q}])  = sample_pred_std(:,q);
    end
    writetable(T_samp, fullfile(out_dir,'predictions_sample_level_multi.csv'));
    T_metrics = table(response_names(:), R2_sample(:), RMSE_sample(:), 'VariableNames',{'target','R2_sample','RMSE_sample'});
    writetable(T_metrics, fullfile(out_dir,'metrics_by_target_cv.csv'));

    % 全数据系数（标准化域）
    if zscore_X, muX_all=mean(X_rep,1); sigX_all=std(X_rep,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(X_rep-muX_all)./sigX_all; else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=X_rep; end
    if zscore_Y, muY_all=mean(Y_rep,1); sigY_all=std(Y_rep,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(Y_rep-muY_all)./sigY_all; else, muY_all=zeros(1,Q); sigY_all=ones(1,Q); Yz_all=Y_rep; end
    [~,~,~,~,beta_all] = plsregress(Xz_all, Yz_all, bestLV);
    coefZ = beta_all(2:end,:); coef_tbl = table(wl_roi, coefZ); 
    coef_tbl.Properties.VariableNames=[{'wavelength_nm'}, strcat('betaZ_',response_names)];
    writetable(coef_tbl, fullfile(out_dir,'plsr_coefficients_roi_multi.csv'));

    model = struct();
    model.betaZ=beta_all; model.nLV=bestLV; model.zscore_X=zscore_X; model.zscore_Y=zscore_Y;
    model.muX=muX_all; model.sigX=sigX_all; model.muY=muY_all; model.sigY=sigY_all;
    model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
    model.use_lines_only=use_lines_only; model.lines_nm=exist('lines_nm','var')*lines_nm(:); model.win_nm=win_nm;
    save(fullfile(out_dir,'plsr_model_multi_xlsx_cv.mat'),'-struct','model','-v7.3');
end

fprintf('\n[DONE] 多指标 PLSR 完成（模式：%s）。输出目录：%s\n', upper(eval_mode), out_dir);
if open_output_dir && ispc, try, system(sprintf('explorer \"%s\"', out_dir)); end, end

%% ======= 局部小函数（放在脚本末尾，便于直接运行） =======
function y = ternary(cond, a, b), if cond, y=a; else, y=b; end
end
function [slope, intercept] = wlinfit(x, y, w)
    x = x(:); y = y(:);
    if nargin<3 || isempty(w), w = ones(size(x)); end
    X = [x ones(size(x))];
    b = lscov(X, y, w);
    slope = b(1); intercept = b(2);
end
function [y_true_s, y_mean_s, y_std_s] = agg_by_sample(Y_rep, Yhat_rep, G_rep, sample_set, mask_set)
    % 假设：Yhat_rep 的行数 = length(G_rep)（本脚本已在调用前扩展 *_full 保证）
    if isempty(sample_set)
        y_true_s = zeros(0,size(Y_rep,2)); y_mean_s = y_true_s; y_std_s = y_true_s; return;
    end
    Q = size(Y_rep,2);
    y_true_s = zeros(numel(sample_set), Q);
    y_mean_s = zeros(numel(sample_set), Q);
    y_std_s  = zeros(numel(sample_set), Q);
    for i=1:numel(sample_set)
        sid = sample_set(i);
        ii  = (G_rep==sid) & mask_set;          % 逻辑索引与 Yhat_rep 同长度
        y_true_s(i,:) = Y_rep(find(ii,1,'first'),:);
        y_mean_s(i,:) = mean(Yhat_rep(ii,:),1);
        y_std_s(i,:)  = std( Yhat_rep(ii,:),0,1);
    end
end
