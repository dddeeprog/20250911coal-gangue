%% PLSR（多指标；Excel 浓度；使用 MAT 中所有样品与复测；样品分组CV）
% 模式：PLS1（逐指标建模，默认，能打破“R² 全相同”）；PLS2（多响应联合）。
% 评估：样品级与复测级均计算；作图导出；内置自检避免列映射/共线导致的假象。

clear; clc; close all; rng(2025,'twister');

%% ===== 配置 =====
MODE = 'PLS1';              % 'PLS1'（推荐）或 'PLS2'
Y_WHITEN = false;            % 仅对 PLS2 可用：对 Y 做训练折内白化，缓解输出共线
Kfold_grp_max = 6;           % 样品分组CV的折数上限
maxLV_cap    = 15;           % LV 最大上限
zscore_X = true; zscore_Y = true;

%% ===== 路径（绝对路径）与输出目录 =====
ave_mat     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\110A110B.mat'; % ← 你的 MAT
excel_xlsx  = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\targets.xlsx'; % ← 你的 Excel
excel_sheet = '';   % 指定工作表名或留空自动
out_dir     = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\plsr_out_multi_xlsx';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 指标（列顺序）与同义列名 =====
response_names = {'TotalS','Ash','Volatile','HHV','C','H','N'};  % 目标顺序
synonyms = containers.Map('KeyType','char','ValueType','any');
synonyms('TotalS')  = lower(string({'TotalS','Sulfur','Total_S','S_total','S'}));
synonyms('Ash')     = lower(string({'Ash','AshContent','Ash_Content'}));
synonyms('Volatile')= lower(string({'Volatile','VM','Vol','VolatileMatter'}));
synonyms('HHV')     = lower(string({'HHV','GCV','CalorificValue','HighHeatingValue'}));
synonyms('C')       = lower(string({'C','Carbon'}));
synonyms('H')       = lower(string({'H','Hydrogen'}));
synonyms('N')       = lower(string({'N','Nitrogen'}));

%% ===== 读取 MAT（自动识别波长轴与样品矩阵） =====
S = load(ave_mat); fn = fieldnames(S);
axis_candidates = {'wavelengths','wavelength','lambda','wavelenths'}; % 包含常见拼写
wl = []; axis_name = '';
for i = 1:numel(axis_candidates)
    nm = axis_candidates{i};
    if isfield(S,nm) && isnumeric(S.(nm)) && isvector(S.(nm))
        wl = S.(nm)(:); axis_name = nm; break;
    end
end
if isempty(wl), error('未找到波长轴变量（尝试 wavelengths / wavelength / lambda / wavelenths）'); end
M = numel(wl);

spec_info = struct('name',{},'transposed',{});
for i = 1:numel(fn)
    nm = fn{i}; if strcmpi(nm,axis_name), continue; end
    val = S.(nm);
    if isnumeric(val) && ndims(val)==2
        [h,w] = size(val);
        if h==M && w>=1
            spec_info(end+1)=struct('name',nm,'transposed',false); %#ok<SAGROW>
        elseif w==M && h>=1
            spec_info(end+1)=struct('name',nm,'transposed',true); %#ok<SAGROW>
        end
    end
end
if isempty(spec_info), error('MAT 中未发现与波长长度匹配的样品矩阵'); end
% 若名字尾部带数字则按数字排序
nums = nan(numel(spec_info),1);
for i=1:numel(spec_info)
    t = regexp(spec_info(i).name,'(\d+)$','tokens','once');
    if ~isempty(t), nums(i)=str2double(t{1}); end
end
[~,ord] = sortrows([isnan(nums) nums],[1 2]); spec_info = spec_info(ord);
spec_names = string({spec_info.name}); N = numel(spec_names);
disp('[Info] 识别样品变量顺序：'); disp(spec_names.');

%% ===== 读取 Excel，并与样品对齐为 N×Q 的浓度矩阵 =====
if isempty(excel_sheet)
    T = readtable(excel_xlsx);
else
    T = readtable(excel_xlsx,'Sheet',excel_sheet);
end
varsL = lower(string(T.Properties.VariableNames));
% 识别样品键列（优先：name/spec/sample，其次：idx/id）
key_name_candidates = lower(["sample","name","spec","sample_name"]);
key_idx_candidates  = lower(["idx","id","sample_idx","order"]);
key_name_col = find(ismember(varsL,key_name_candidates),1);
key_idx_col  = find(ismember(varsL,key_idx_candidates),1);
if isempty(key_name_col) && isempty(key_idx_col)
    error('Excel 需要包含样品标识列（名称列：sample/name/spec 或 索引列：idx/id）。');
end
% 为每个 response 找对应列（大小写不敏感、支持同义名）
Q = numel(response_names);
col_idx = nan(1,Q);
for q=1:Q
    keys = synonyms(response_names{q}); % string 数组（已 lower）
    hitc = NaN;
    for c = 1:numel(varsL)
        if any(strcmpi(varsL(c), string(keys)))
            hitc = c; break;
        end
    end
    if isnan(hitc)
        error('Excel 中未找到指标列：%s（支持同义名：%s）', response_names{q}, strjoin(keys,','));
    end
    col_idx(q) = hitc;
end
% 自检：目标->Excel 列映射唯一性
fprintf("[CHECK] Target -> Excel column mapping");
for q = 1:Q
    fprintf("  %-10s -> %s", response_names{q}, T.Properties.VariableNames{col_idx(q)});
end
if numel(unique(col_idx)) < numel(col_idx)
    ucols = unique(col_idx); dup_msgs = strings(0);
    for uc = ucols(:)'
        qset = find(col_idx==uc); if numel(qset) > 1
            dup_msgs(end+1) = sprintf('%s -> %s', strjoin(response_names(qset),','), T.Properties.VariableNames{uc}); %#ok<SAGROW>
        end
    end
    error('多个指标映射到了同一 Excel 列:%s请改正列名或收紧 synonyms 后重试。', strjoin(dup_msgs, ''));
end
% 构建 N×Q 的浓度矩阵 conc_all，顺序与 spec_names 一致
conc_all = nan(N,Q);
if ~isempty(key_name_col)
    keyvals = string(T{:,key_name_col}); keyvals = strtrim(lower(keyvals));
    specL = strtrim(lower(spec_names));
    for i=1:N
        hit = find(strcmp(specL(i), keyvals), 1, 'first');
        if isempty(hit)
            error('Excel 中未找到与样品变量 "%s" 对应的名称（大小写与空白不敏感）。', spec_names(i));
        end
        conc_all(i,:) = T{hit, col_idx};
    end
else
    idxvals = T{:,key_idx_col};
    if ~isnumeric(idxvals), error('索引列必须为数值。'); end
    for i=1:N
        hit = find(idxvals==i, 1, 'first');
        if isempty(hit)
            error('Excel 中未找到样品索引 = %d 的行。', i);
        end
        conc_all(i,:) = T{hit, col_idx};
    end
end
% 标签相关矩阵（样品级）
try
    disp('[CHECK] label列两两相关系数 (conc_all)：'); disp(corr(conc_all,'rows','pairwise'));
catch, warning('无法计算相关矩阵'); end
% 缺失检查
if any(isnan(conc_all(:)))
    warning('检测到 conc_all 中存在 NaN，请检查 Excel 是否缺数据。');
end

%% ===== ROI（全谱 or 谱线窗口） =====
use_lines_only = false;                 % 如需窗口，改为 true 并设置 lines_nm/win_nm
lines_nm = [394.4, 396.15, 279.55, 280.27, 285.21, 373.49, 373.74];
win_nm   = 0.30;
if use_lines_only
    mask = false(size(wl));
    for L = lines_nm, mask = mask | (abs(wl - L) <= win_nm); end
    if ~any(mask), error('谱线窗口掩码为空；请检查 lines_nm/win_nm'); end
else
    mask = true(size(wl));
end
wl_roi = wl(mask); P = nnz(mask);  % 特征维

%% ===== 展开复测为“观测级样本” =====
X_rep = [];            % [n_obs × P]
Y_rep = [];            % [n_obs × Q]
G_rep = [];            % [n_obs × 1] 样品编号
R_idx = [];            % [n_obs × 1] 复测序号
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
fprintf('[Info] 观测=%d | 样品=%d | 特征=%d | 指标=%d', n_obs, S_used, P, numel(response_names));

%% ===== 分组CV 折分 =====
Kfold_grp = min(Kfold_grp_max, S_used); maxLV = min([maxLV_cap,n_obs-1,P]); rng(2025);
perm = unique_samp(randperm(S_used));
fold_of_sample = zeros(S_used,1); for i=1:S_used, fold_of_sample(i) = mod(i-1,Kfold_grp)+1; end
map = containers.Map(num2cell(perm), num2cell(fold_of_sample));
F_rep = zeros(n_obs,1); for i=1:n_obs, F_rep(i) = map(G_rep(i)); end

%% ===== 训练与CV预测 =====
yhat_cv = nan(n_obs, Q); bestLVs = nan(1,Q);

switch upper(MODE)
case 'PLS2'
    % ---------- PLS2：联合建模（可选Y白化） ----------
    MSEz_mean = nan(maxLV,1);
    for lv = 1:maxLV
        SSEz = zeros(1,Q); CNT = zeros(1,Q);
        for f = 1:Kfold_grp
            tr = F_rep~=f; te = ~tr;
            XT = X_rep(tr,:); XE = X_rep(te,:); YT = Y_rep(tr,:); YE = Y_rep(te,:);
            % 标准化
            if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
            if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; YTz=(YT-muY)./sigY; YE_z=(YE-muY)./sigY; else, muY=zeros(1,Q); sigY=ones(1,Q); YTz=YT; YE_z=YE; end
            if Y_WHITEN
                C = cov(YTz,1); [V,D]=eig((C+C')/2); D = max(D,0); W = V*diag(1./sqrt(diag(D)+eps))*V';
                YTz = YTz*W; YE_z = YE_z*W; UnW = V*diag(sqrt(diag(D)+eps))*V'; % 反白化
            else, W=[]; UnW=[]; end
            [~,~,~,~,beta] = plsregress(XTz, YTz, lv);
            Yhat_z = [ones(sum(te),1) XEz] * beta;
            if Y_WHITEN, Yhat_z = Yhat_z*UnW; end
            Rz = Yhat_z - YE_z; SSEz = SSEz + sum(Rz.^2,1); CNT = CNT + sum(~isnan(Rz),1);
        end
        MSEz_mean(lv) = mean(SSEz ./ max(CNT,1));
    end
    [best_mse, bestLV] = min(MSEz_mean); bestLVs(:) = bestLV;
    fprintf('[CV-group][PLS2] bestLV=%d | mean MSE_z=%.6g | folds=%d', bestLV, best_mse, Kfold_grp);
    % 复算CV预测（原单位）
    for f = 1:Kfold_grp
        tr = F_rep~=f; te = ~tr;
        XT = X_rep(tr,:); XE = X_rep(te,:); YT = Y_rep(tr,:);
        if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
        if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; else, muY=zeros(1,Q); sigY=ones(1,Q); end
        if Y_WHITEN, C = cov((YT-muY)./sigY,1); [V,D]=eig((C+C')/2); D=max(D,0); W=V*diag(1./sqrt(diag(D)+eps))*V'; UnW=V*diag(sqrt(diag(D)+eps))*V'; else, W=[]; UnW=[]; end
        [~,~,~,~,beta] = plsregress(XTz, ((YT - muY)./sigY)*(isempty(W)+W), bestLV);
        Yhat_z = [ones(sum(te),1) XEz] * beta; if Y_WHITEN, Yhat_z = Yhat_z*UnW; end
        yhat_cv(te,:) = Yhat_z.*sigY + muY;
    end

case 'PLS1'
    % ---------- PLS1：逐指标建模（推荐） ----------
    for q = 1:Q
        MSEz = nan(maxLV,1);
        for lv = 1:maxLV
            SSE=0; CNT=0;
            for f = 1:Kfold_grp
                tr = F_rep~=f; te = ~tr;
                XT = X_rep(tr,:); XE = X_rep(te,:); yT = Y_rep(tr,q); yE = Y_rep(te,q);
                if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
                if zscore_Y, muy=mean(yT); sigy=std(yT); if sigy==0, sigy=eps; end; yTz=(yT-muy)/sigy; yEz=(yE-muy)/sigy; else, muy=0; sigy=1; yTz=yT; yEz=yE; end
                [~,~,~,~,beta] = plsregress(XTz, yTz, lv);
                yhat_z = [ones(sum(te),1) XEz]*beta; r = yhat_z - yEz; SSE = SSE + nansum(r.^2); CNT = CNT + sum(~isnan(r));
            end
            MSEz(lv) = SSE/max(CNT,1);
        end
        [~, bestLV] = min(MSEz); bestLVs(q)=bestLV;
        for f = 1:Kfold_grp
            tr = F_rep~=f; te = ~tr;
            XT = X_rep(tr,:); XE = X_rep(te,:); yT = Y_rep(tr,q);
            if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
            if zscore_Y, muy=mean(yT); sigy=std(yT); if sigy==0, sigy=eps; end; else, muy=0; sigy=1; end
            [~,~,~,~,beta] = plsregress(XTz, (yT-muy)/sigy, bestLV);
            yhat_z = [ones(sum(te),1) XEz]*beta; yhat_cv(te,q) = yhat_z*sigy + muy;
        end
    end
    fprintf('[CV-group][PLS1] bestLV per target = %s', mat2str(bestLVs));
otherwise
    error('未知 MODE = %s', MODE);
end

%% ===== 全数据训练最终模型（保存部署参数） =====
if zscore_X, muX_all=mean(X_rep,1); sigX_all=std(X_rep,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(X_rep-muX_all)./sigX_all; else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=X_rep; end
muY_all = zeros(1,Q); sigY_all = ones(1,Q);
if strcmpi(MODE,'PLS2')
    if zscore_Y, muY_all=mean(Y_rep,1); sigY_all=std(Y_rep,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(Y_rep-muY_all)./sigY_all; else, Yz_all=Y_rep; end
    [~,~,~,~,beta_all] = plsregress(Xz_all, Yz_all, max(bestLVs));
else % PLS1 保存为列拼接
    beta_all = zeros(P+1,Q);
    for q=1:Q
        y=Y_rep(:,q); if zscore_Y, muY_all(q)=mean(y); sigY_all(q)=std(y); if sigY_all(q)==0, sigY_all(q)=eps; end; yz=(y-muY_all(q))/sigY_all(q); else, yz=y; end
        [~,~,~,~,beta_q] = plsregress(Xz_all, yz, bestLVs(q)); beta_all(:,q)=beta_q;
    end
end

%% ===== 指标评估：样品级 + 复测级 =====
% 样品级聚合（CV 预测）
sample_true = zeros(N,Q); sample_pred_mean=zeros(N,Q); sample_pred_std=zeros(N,Q); sample_nrep=zeros(N,1);
for t=1:N
    ii = (G_rep==t);
    sample_true(t,:)      = Y_rep(find(ii,1,'first'),:);
    sample_pred_mean(t,:) = mean(yhat_cv(ii,:),1);
    sample_pred_std(t,:)  = std(yhat_cv(ii,:),0,1);
    sample_nrep(t)        = sum(ii);
end
R2_sample=zeros(1,Q); RMSE_sample=zeros(1,Q);
for q=1:Q
    y=sample_true(:,q); yh=sample_pred_mean(:,q);
    SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
    R2_sample(q)=1-SSE/SST; RMSE_sample(q)=sqrt(mean((y-yh).^2));
end
% 复测级评估（不聚合）
R2_rep=zeros(1,Q); RMSE_rep=zeros(1,Q);
for q=1:Q
    y=Y_rep(:,q); yh=yhat_cv(:,q);
    SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
    R2_rep(q)=1-SSE/SST; RMSE_rep(q)=sqrt(mean((y-yh).^2));
end

% 诊断输出
try, disp('[CHECK] 预测矩阵 yhat_cv 两两相关系数：'); disp(corr(yhat_cv,'rows','pairwise')); catch, end
fprintf('[CHECK] per-target summary (sample-level): R2 / RMSE | bestLV');
for q=1:Q, fprintf('%-10s | R2_s=%.6f RMSE_s=%.4g | R2_r=%.6f RMSE_r=%.4g | LV=%d', ...
        response_names{q}, R2_sample(q), RMSE_sample(q), R2_rep(q), RMSE_rep(q), bestLVs(q)); end

%% ===== 可视化与导出 =====
% LV 选择曲线（仅 PLS2 有整体曲线；PLS1 可另存 per-target 曲线，如需再加）
if strcmpi(MODE,'PLS2')
    fig1=figure('Name','PLSR_groupCV_multi','Position',[80 80 560 380]);
    plot(1:numel(MSEz_mean), MSEz_mean, '-o'); grid on; xlabel('LV'); ylabel('mean MSE_z'); title('分组CV（样品）- 多指标');
    saveas(fig1, fullfile(out_dir,'plsr_groupCV_multi.png'));
end
% 每指标的样品级误差棒
for q=1:Q
    fig=figure('Name',['Err_',response_names{q}],'Position',[660 80 560 440]);
    errorbar(sample_true(:,q), sample_pred_mean(:,q), sample_pred_std(:,q), 'o','LineWidth',1.4,'MarkerSize',6,'CapSize',6);
    hold on; grid on; box on;
    xymin=min([sample_true(:,q); sample_pred_mean(:,q)-sample_pred_std(:,q)]);
    xymax=max([sample_true(:,q); sample_pred_mean(:,q)+sample_pred_std(:,q)]);
    plot([xymin xymax],[xymin xymax],'k--');
    xlabel(['真实 ',response_names{q}]); ylabel(['预测 ',response_names{q},'（均值±STD）']);
    ttlLV = strcmpi(MODE,'PLS2')*max(bestLVs) + strcmpi(MODE,'PLS1')*bestLVs(q);
    title(sprintf('%s | R^2=%.4f (sample), RMSE=%.4g | LV=%d', response_names{q}, R2_sample(q), RMSE_sample(q), ttlLV));
    saveas(fig, fullfile(out_dir, sprintf('errorbar_sample_%s.png', response_names{q})));
end
% 导出复测级/样品级/指标汇总
T_rep = table(G_rep, R_idx, 'VariableNames',{'sample_idx','rep_idx'});
for q=1:Q
    T_rep.(['true_',response_names{q}])   = Y_rep(:,q);
    T_rep.(['predCV_',response_names{q}]) = yhat_cv(:,q);
end
writetable(T_rep, fullfile(out_dir,'predictions_replicate_level_multi.csv'));
T_samp = table((1:N).', sample_nrep, 'VariableNames',{'sample_idx','n_reps'});
for q=1:Q
    T_samp.(['true_',response_names{q}])     = sample_true(:,q);
    T_samp.(['predMean_',response_names{q}]) = sample_pred_mean(:,q);
    T_samp.(['predSTD_',response_names{q}])  = sample_pred_std(:,q);
end
writetable(T_samp, fullfile(out_dir,'predictions_sample_level_multi.csv'));
T_metrics = table(response_names(:), bestLVs(:), R2_sample(:), RMSE_sample(:), R2_rep(:), RMSE_rep(:), ...
    'VariableNames',{'target','bestLV','R2_sample','RMSE_sample','R2_rep','RMSE_rep'});
writetable(T_metrics, fullfile(out_dir,'metrics_by_target.csv'));
% 系数谱（标准化域）与模型保存（便于部署）
coefZ = beta_all(2:end,:);  % P×Q（PLS1/PLS2 通用：列=指标）
colnames = strcat('betaZ_', response_names); colnames = matlab.lang.makeValidName(colnames);
coef_tbl = [ table(wl_roi, 'VariableNames', {'wavelength_nm'}), array2table(coefZ, 'VariableNames', colnames) ];
writetable(coef_tbl, fullfile(out_dir,'plsr_coefficients_roi_multi.csv'));
model=struct();
model.mode=MODE; model.betaZ=beta_all; model.nLV=bestLVs; model.zscore_X=zscore_X; model.zscore_Y=zscore_Y;
model.muX=muX_all; model.sigX=sigX_all; model.muY=muY_all; model.sigY=sigY_all;
model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
model.use_lines_only=use_lines_only; model.lines_nm=lines_nm(:); model.win_nm=win_nm; model.Y_WHITEN=Y_WHITEN;
save(fullfile(out_dir,'plsr_model_multi_xlsx.mat'),'-struct','model','-v7.3');

fprintf('[DONE] 多指标 %s 完成。输出目录：%s', MODE, out_dir);
