%% PLSR（多指标；从 Excel 读取各组分浓度；使用 MAT 中所有样品与复测；分组CV）
% 支持指标：默认 {'TotalS','Ash','Volatile','HHV','C','H','N'}（可改）。
% 数据来源：
%   1) 光谱：ave.mat（自动识别波长轴与样品矩阵；样品矩阵可为 MxK 或 KxM）
%   2) 浓度：Excel（支持按样品名或样品索引对齐；列名大小写不敏感、支持常见同义名）
% 评估：按“样品分组”的交叉验证选择 LV；样品级均值±STD 误差棒；导出多指标结果与模型。

clear; clc; close all; rng(2025,'twister');

%% ===== 路径（绝对路径）与输出目录 =====
ave_mat   = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\110A110B.mat'; % ← 你的 MAT
excel_xlsx= 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\targets.xlsx';                                % ← 你的 Excel
excel_sheet = '';   % 指定工作表名或留空自动
out_dir   = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B\plsr_out_multi_xlsx';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 指标（列顺序）与同义列名 =====
response_names = {'TotalS','Ash','Volatile','HHV','C','H','N'};  % 目标顺序
synonyms = containers.Map();
synonyms('TotalS') = lower({'TotalS','S','Sulfur','Total_S','S_total'});
synonyms('Ash')    = lower({'Ash','AshContent','Ash_Content'});
synonyms('Volatile')=lower({'Volatile','VM','Vol','VolatileMatter'});
synonyms('HHV')    = lower({'HHV','GCV','CalorificValue','HighHeatingValue'});
synonyms('C')      = lower({'C','Carbon'});
synonyms('H')      = lower({'H','Hydrogen'});
synonyms('N')      = lower({'N','Nitrogen'});

%% ===== 读取 MAT（自动识别波长轴与样品矩阵） =====
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
        elseif w==M && h>=1, spec_info(end+1)=struct('name',nm,'transposed',true); %#ok<SAGROW>
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
    keys = synonyms(response_names{q});
    for c = 1:numel(varsL)
        if any(strcmpi(varsL(c), string(keys)))
            col_idx(q) = c; break;
        end
    end
    if isnan(col_idx(q))
        error('Excel 中未找到指标列：%s（支持同义名：%s）', response_names{q}, strjoin(keys,','));
    end
end

% 构建 N×Q 的浓度矩阵 conc_all，顺序与 spec_names 一致
conc_all = nan(N,Q);
if ~isempty(key_name_col)
    % 名称匹配（大小写不敏感；去掉两端空格）
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
    % 索引匹配（假设 1..N）
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

% 简要检查缺失
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
fprintf('[Info] 观测=%d | 样品=%d | 特征=%d | 指标=%d\n', n_obs, S_used, P, numel(response_names));

%% ===== 标准化与分组CV选择 LV（Y 标准化以平均MSE做目标） =====
zscore_X = true; zscore_Y = true; Kfold_grp = min(6,S_used); maxLV = min([15,n_obs-1,P]); rng(2025);
perm = unique_samp(randperm(S_used));
fold_of_sample = zeros(S_used,1); for i=1:S_used, fold_of_sample(i) = mod(i-1,Kfold_grp)+1; end
map = containers.Map(num2cell(perm), num2cell(fold_of_sample));
F_rep = zeros(n_obs,1); for i=1:n_obs, F_rep(i) = map(G_rep(i)); end

MSEz_mean = nan(maxLV,1);
for lv = 1:maxLV
    SSEz = zeros(1,numel(response_names)); CNT = zeros(1,numel(response_names));
    for f = 1:Kfold_grp
        tr = F_rep~=f; te = ~tr;
        XT = X_rep(tr,:); XE = X_rep(te,:);
        YT = Y_rep(tr,:); YE = Y_rep(te,:);
        % 标准化
        if zscore_X
            muX = mean(XT,1); sigX = std(XT,[],1); sigX(sigX==0)=eps;
            XTz = (XT - muX)./sigX; XEz = (XE - muX)./sigX;
        else, XTz = XT; XEz = XE; end
        if zscore_Y
            muY = mean(YT,1); sigY = std(YT,[],1); sigY(sigY==0)=eps;
            YTz = (YT - muY)./sigY; YE_z = (YE - muY)./sigY;
        else, muY=zeros(1,size(YT,2)); sigY=ones(1,size(YT,2)); YTz=YT; YE_z=YE; end
        % 训练/预测
        [~,~,~,~,beta] = plsregress(XTz, YTz, lv);
        Yhat_z = [ones(sum(te),1) XEz] * beta;
        Rz = Yhat_z - YE_z; SSEz = SSEz + sum(Rz.^2,1); CNT = CNT + sum(~isnan(Rz),1);
    end
    MSEz_mean(lv) = mean(SSEz ./ max(CNT,1));
end
[best_mse, bestLV] = min(MSEz_mean);
fprintf('[CV-group] bestLV=%d | mean MSE_z=%.6g | folds=%d\n', bestLV, best_mse, Kfold_grp);

% 复算一次得到每条复测的 CV 预测（原单位）
yhat_cv = nan(n_obs, numel(response_names));
for f = 1:Kfold_grp
    tr = F_rep~=f; te = ~tr;
    XT = X_rep(tr,:); XE = X_rep(te,:); YT = Y_rep(tr,:);
    if zscore_X
        muX = mean(XT,1); sigX = std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX;
    else, XTz=XT; XEz=XE; end
    if zscore_Y
        muY = mean(YT,1); sigY = std(YT,[],1); sigY(sigY==0)=eps;
    else, muY=zeros(1,size(YT,2)); sigY=ones(1,size(YT,2)); end
    [~,~,~,~,beta] = plsregress(XTz, (YT - muY)./sigY, bestLV);
    Yhat_z = [ones(sum(te),1) XEz] * beta; yhat_cv(te,:) = Yhat_z.*sigY + muY;
end

%% ===== 全数据训练最终模型（标准化域系数），并还原预测 =====
if zscore_X
    muX_all = mean(X_rep,1); sigX_all = std(X_rep,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(X_rep-muX_all)./sigX_all;
else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=X_rep; end
if zscore_Y
    muY_all = mean(Y_rep,1); sigY_all = std(Y_rep,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(Y_rep-muY_all)./sigY_all;
else, muY_all=zeros(1,numel(response_names)); sigY_all=ones(1,numel(response_names)); Yz_all=Y_rep; end
[~,~,~,~,beta_all] = plsregress(Xz_all, Yz_all, bestLV);
Yhat_fit_all = [ones(n_obs,1) Xz_all]*beta_all.*sigY_all + muY_all;  %#ok<NASGU>

%% ===== 样品级聚合（CV 预测）：均值±STD 与每指标 R²/RMSE =====
Q = numel(response_names);
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

%% ===== 可视化与导出 =====
% CV 选 LV 曲线
fig1=figure('Name','PLSR_groupCV_multi','Position',[80 80 560 380]);
plot(1:numel(MSEz_mean), MSEz_mean, '-o'); grid on; xlabel('LV'); ylabel('mean MSE_z'); title('分组CV（样品）- 多指标');
saveas(fig1, fullfile(out_dir,'plsr_groupCV_multi.png'));

% 每指标的样品级误差棒
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

T_metrics = table(response_names(:), R2_sample(:), RMSE_sample(:), 'VariableNames',{'target','R2_sample','RMSE_sample'});
writetable(T_metrics, fullfile(out_dir,'metrics_by_target.csv'));

% 系数谱（标准化域）与模型保存（便于部署）
coefZ = beta_all(2:end,:);  % P×Q
coef_tbl = table(wl_roi, coefZ); coef_tbl.Properties.VariableNames=[{'wavelength_nm'}, strcat('betaZ_',response_names)];
writetable(coef_tbl, fullfile(out_dir,'plsr_coefficients_roi_multi.csv'));

model=struct();
model.betaZ=beta_all; model.nLV=bestLV; model.zscore_X=true; model.zscore_Y=true;
model.muX=muX_all; model.sigX=sigX_all; model.muY=muY_all; model.sigY=sigY_all;
model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
model.use_lines_only=false; model.lines_nm=lines_nm(:); model.win_nm=win_nm;
save(fullfile(out_dir,'plsr_model_multi_xlsx.mat'),'-struct','model','-v7.3');

fprintf('\n[DONE] 多指标 PLSR（Excel 浓度）完成。输出目录：%s\n', out_dir);
