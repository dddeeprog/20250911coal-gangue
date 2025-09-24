%% PLSR（多指标；18样品；样品分组的嵌套CV + 预留外测；含 VIP 与 T^2/Q 图）
% - 外测样品（ext_test_list）完全不参与任何折的训练与选参（无信息泄漏）。
% - 外层：在“训练池”样品上做 K 折（样品分组）。
% - 内层：在外层训练样品上再做分组CV扫描 LV，选出每折 bestLV。
% - 汇总：得到训练样品的嵌套CV折外预测（蓝色）；
% - 最终：用整个训练池重训（bestLV_final），对外测样品做一次性评估（红色）。
% - 新增：VIP（总体与逐指标）与 Hotelling T^2 / Q 残差（训练池 vs 外测）图与表。

clear; clc; close all; rng(2025,'twister');

%% ===== 路径与输出 =====
ave_mat     = 'C:\\Users\\TomatoK\\Desktop\\20250911coal&gangue\\gangue\\spec\\ave_5\\all\\all.mat';
excel_xlsx  = 'C:\\Users\\TomatoK\\Desktop\\20250911coal&gangue\\gangue\\spec\\ave_5\\all\\targets\\targets_18.xlsx';
excel_sheet = '';   % 指定表名或留空自动
out_dir     = 'C:\\Users\\TomatoK\\Desktop\\20250911coal&gangue\\gangue\\spec\\ave_5\\all\\plsr_out_nestedCV18_extHoldout';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% ===== 配置：固定18样品 + 嵌套CV + 外测预留 =====
FORCE_N      = 18;       % 强制只用 18 个样品（按变量名尾号排序后的前18个）
% —— 外测样品索引（基于上述排序后的 1..18 序号）。请按需修改：
ext_test_list = [3 4 9 10 15 16];   % 示例：预留 6 个外测样品
% 嵌套CV参数（仅在“训练池=1..18 \\ ext_test_list”上进行）
outerK       = 6;        % 外层折数（训练池样品按折分组；建议 3~6）
innerK       = 5;        % 内层折数（每个外层训练集再分折）
maxLV_cap    = 15;       % LV 上限硬阈值
rng_seed     = 2025;     % 随机种子

% 预处理与 ROI
zscore_X = true; zscore_Y = true;
use_lines_only = false;         % false=全谱；true=只用谱线窗口
lines_nm   = [394.4, 396.15, 279.55, 280.27, 285.21, 373.49, 373.74];
win_nm     = 0.30;              % 谱线半窗宽（nm）

%% ===== 目标与同义名 =====
response_names = {'TotalS','Ash','Volatile','HHV','C','H','N'}; Qnum = numel(response_names);
synonyms = containers.Map();
synonyms('TotalS')  = lower({'TotalS','S','Sulfur','Total_S','S_total'});
synonyms('Ash')     = lower({'Ash','AshContent','Ash_Content'});
synonyms('Volatile')= lower({'Volatile','VM','Vol','VolatileMatter'});
synonyms('HHV')     = lower({'HHV','GCV','CalorificValue','HighHeatingValue'});
synonyms('C')       = lower({'C','Carbon'});
synonyms('H')       = lower({'H','Hydrogen'});
synonyms('N')       = lower({'N','Nitrogen'});

%% ===== 读取 MAT（波长与样品矩阵） =====
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
% 按尾号排序并截取前18个
nums = nan(numel(spec_info),1);
for i=1:numel(spec_info)
    t = regexp(spec_info(i).name,'(\d+)$','tokens','once');
    if ~isempty(t), nums(i)=str2double(t{1}); end
end
[~,ord] = sortrows([isnan(nums) nums],[1 2]); spec_info = spec_info(ord);
if numel(spec_info) < FORCE_N, error('仅识别到 %d 个样品，少于 18。', numel(spec_info)); end
spec_info = spec_info(1:FORCE_N);
spec_names = string({spec_info.name}); N = numel(spec_names);
fprintf('[Info] 使用样品：\n'); disp(spec_names.');

%% ===== 读取 Excel 并对齐 N×Q 标签 =====
if isempty(excel_sheet), T = readtable(excel_xlsx);
else,                    T = readtable(excel_xlsx,'Sheet',excel_sheet);
end
varsL = lower(string(T.Properties.VariableNames));
key_name_candidates = lower(["sample","name","spec","sample_name"]);
key_idx_candidates  = lower(["idx","id","sample_idx","order"]);
key_name_col = find(ismember(varsL,key_name_candidates),1);
key_idx_col  = find(ismember(varsL,key_idx_candidates),1);
col_idx = nan(1,Qnum);
for q=1:Qnum
    keys = synonyms(response_names{q});
    hitc = find(ismember(varsL, string(keys)), 1, 'first');
    if ~isempty(hitc), col_idx(q)=hitc; else, error('Excel 缺少指标列：%s', response_names{q}); end
end
conc_all = nan(N,Qnum);
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
    conc_all = T{1:N, col_idx};
end
if any(isnan(conc_all(:))), warning('conc_all 中有 NaN，请检查 Excel。'); end

%% ===== ROI（全谱或谱线窗口） =====
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
n_obs = size(X_rep,1); fprintf('[Info] 观测=%d | 样品=%d | 特征=%d | 指标=%d\n', n_obs, N, P, Qnum);

%% ===== 划分：外测样品 & 训练池样品 =====
ext_samples = unique(ext_test_list(:)');
assert(all(ismember(ext_samples, 1:N)), 'ext_test_list 必须是 1..%d 范围内的样品索引', N);
train_pool_samples = setdiff(1:N, ext_samples);
if numel(train_pool_samples) < 4, error('训练池样品过少（<4），请减少外测保留数或增加样品。'); end

is_ext  = ismember(G_rep, ext_samples);
is_pool = ismember(G_rep, train_pool_samples);

%% ===== 外层：仅对训练池做样品分组 K 折 =====
rng(rng_seed,'twister');
outerK_eff = min(outerK, numel(train_pool_samples)); if outerK_eff < 2, outerK_eff = 2; end
perm_samples = train_pool_samples(randperm(numel(train_pool_samples)));
fold_of_sample_outer = containers.Map('KeyType','int32','ValueType','int32');
for i=1:numel(perm_samples), fold_of_sample_outer(perm_samples(i)) = mod(i-1, outerK_eff)+1; end

% 结果容器（仅对训练池样品做外层折外预测）
yhat_outercv = nan(n_obs, Qnum);        % 每条训练池观测在其外层验证折被预测一次
bestLV_outer  = nan(outerK_eff,1);

for f_outer = 1:outerK_eff
    % 当前外层的验证样品（均来自训练池）
    val_samples = train_pool_samples(arrayfun(@(s) fold_of_sample_outer(s)==f_outer, train_pool_samples));
    trn_samples = setdiff(train_pool_samples, val_samples);

    is_outer_tr  = ismember(G_rep, trn_samples);
    is_outer_val = ismember(G_rep, val_samples);

    % —— 内层：在 trn_samples 上分组CV选 LV ——
    innerS = numel(trn_samples);
    innerK_eff = min(innerK, innerS); if innerK_eff < 2, innerK_eff = 2; end
    perm_in = trn_samples(randperm(innerS));
    fold_of_sample_inner = containers.Map('KeyType','int32','ValueType','int32');
    for i=1:innerS, fold_of_sample_inner(perm_in(i)) = mod(i-1, innerK_eff)+1; end

    F_rep_inner = nan(n_obs,1);
    for i=1:n_obs
        if is_outer_tr(i)
            F_rep_inner(i) = fold_of_sample_inner(G_rep(i));
        end
    end

    XT_all = X_rep(is_outer_tr,:); YT_all = Y_rep(is_outer_tr,:);
    maxLV = min([maxLV_cap, size(XT_all,1)-1, P]); if maxLV < 1, maxLV = 1; end

    MSEz_mean = nan(maxLV,1);
    for lv = 1:maxLV
        SSEz = zeros(1,Qnum); CNT = zeros(1,Qnum);
        for fi = 1:innerK_eff
            tr = is_outer_tr & (F_rep_inner~=fi); te = is_outer_tr & (F_rep_inner==fi);
            if ~any(tr) || ~any(te), continue; end
            XT = X_rep(tr,:); XE = X_rep(te,:);
            YT = Y_rep(tr,:); YE = Y_rep(te,:);
            if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTz=(XT-muX)./sigX; XEz=(XE-muX)./sigX; else, XTz=XT; XEz=XE; end
            if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; YTz=(YT-muY)./sigY; YE_z=(YE-muY)./sigY; else, muY=zeros(1,Qnum); sigY=ones(1,Qnum); YTz=YT; YE_z=YE; end
            [~,~,~,~,beta] = plsregress(XTz, YTz, lv);
            Yhat_z = [ones(sum(te),1) XEz] * beta;
            Rz = Yhat_z - YE_z; SSEz = SSEz + sum(Rz.^2,1); CNT = CNT + sum(~isnan(Rz),1);
        end
        MSEz_mean(lv) = mean(SSEz ./ max(CNT,1));
    end
    [~, bestLV] = min(MSEz_mean);
    bestLV_outer(f_outer) = bestLV;

    % —— 用 bestLV 在 trn_samples 上重训，并预测 val_samples ——
    XT = XT_all; YT = YT_all; XE = X_rep(is_outer_val,:);
    if zscore_X, muX=mean(XT,1); sigX=std(XT,[],1); sigX(sigX==0)=eps; XTr=(XT-muX)./sigX; XEv=(XE-muX)./sigX; else, muX=zeros(1,P); sigX=ones(1,P); XTr=XT; XEv=XE; end
    if zscore_Y, muY=mean(YT,1); sigY=std(YT,[],1); sigY(sigY==0)=eps; YTr=(YT-muY)./sigY; else, muY=zeros(1,Qnum); sigY=ones(1,Qnum); YTr=YT; end
    [~,~,~,~,beta_o] = plsregress(XTr, YTr, bestLV);
    Yhat_val = [ones(size(XEv,1),1) XEv] * beta_o .* sigY + muY;   % 反标准化

    % 写入该验证折的复测级预测
    yhat_outercv(is_outer_val,:) = Yhat_val;
end

%% ===== 训练（嵌套CV折外）样品级聚合 & 指标 =====
sample_true_tr = zeros(N,Qnum); sample_pred_tr_mean=zeros(N,Qnum);
sample_pred_tr_std=zeros(N,Qnum); sample_nrep_tr=zeros(N,1);
for si = 1:N
    if ~ismember(si, train_pool_samples), continue; end
    idx = (G_rep==si);
    sample_true_tr(si,:)      = Y_rep(find(idx,1,'first'),:);
    sample_pred_tr_mean(si,:) = mean(yhat_outercv(idx,:),1);
    sample_pred_tr_std(si,:)  = std( yhat_outercv(idx,:),0,1);
    sample_nrep_tr(si)        = sum(idx);
end

R2_sample_tr=zeros(1,Qnum); RMSE_sample_tr=zeros(1,Qnum);
R2_rep_tr=zeros(1,Qnum);    RMSE_rep_tr=zeros(1,Qnum);
for q=1:Qnum
    sel = ismember(1:N, train_pool_samples)';
    y  = sample_true_tr(sel,q); yh = sample_pred_tr_mean(sel,q);
    SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2); R2_sample_tr(q)=1-SSE/SST;
    RMSE_sample_tr(q)=sqrt(mean((y-yh).^2));
    % 复测级（仅训练池观测）
    y_rep  = Y_rep(is_pool,q); yh_rep = yhat_outercv(is_pool,q);
    SSEr=sum((y_rep-yh_rep).^2); SStr=sum((y_rep-mean(y_rep)).^2);
    R2_rep_tr(q)=1-SSEr/SStr; RMSE_rep_tr(q)=sqrt(mean((y_rep-yh_rep).^2));
end

%% ===== 选最终 LV 并在“训练池”重训；对外测预测 =====
valid_best = bestLV_outer(~isnan(bestLV_outer));
if isempty(valid_best), error('未能在任何外层折上得到 bestLV。'); end
bestLV_final = max(1, round(median(valid_best)));   % 也可改为 mode(valid_best)

% 训练池观测
XT_all = X_rep(is_pool,:); YT_all = Y_rep(is_pool,:);
if zscore_X, muX_all=mean(XT_all,1); sigX_all=std(XT_all,[],1); sigX_all(sigX_all==0)=eps; Xz_all=(XT_all-muX_all)./sigX_all; else, muX_all=zeros(1,P); sigX_all=ones(1,P); Xz_all=XT_all; end
if zscore_Y, muY_all=mean(YT_all,1); sigY_all=std(YT_all,[],1); sigY_all(sigY_all==0)=eps; Yz_all=(YT_all-muY_all)./sigY_all; else, muY_all=zeros(1,Qnum); sigY_all=ones(1,Qnum); Yz_all=YT_all; end
[XL_all,YL_all,XS_all,YS_all,beta_all,PCTVAR_all,MSE_all,stats_final] = plsregress(Xz_all, Yz_all, bestLV_final); %#ok<ASGLU>

% 外测观测预测（一次性）
XE = X_rep(is_ext,:); Xz_te = (XE - muX_all)./sigX_all;
Yhat_test_rep = [ones(sum(is_ext),1) Xz_te]*beta_all.*sigY_all + muY_all;

% 外测样品级聚合
ext_samples_sorted = sort(ext_samples);
n_te_samp = numel(ext_samples_sorted);
sample_true_te = zeros(n_te_samp,Qnum); sample_pred_te_mean=zeros(n_te_samp,Qnum);
sample_pred_te_std=zeros(n_te_samp,Qnum); sample_nrep_te=zeros(n_te_samp,1);
test_rows = find(is_ext);
for ii=1:n_te_samp
    t   = ext_samples_sorted(ii);
    idx = (G_rep(test_rows)==t);
    rows= test_rows(idx);
    sample_true_te(ii,:)      = Y_rep(rows(1),:);
    sample_pred_te_mean(ii,:) = mean(Yhat_test_rep(idx,:),1);
    sample_pred_te_std(ii,:)  = std( Yhat_test_rep(idx,:),0,1);
    sample_nrep_te(ii)        = sum(idx);
end

% 外测指标
R2_sample_te=zeros(1,Qnum); RMSE_sample_te=zeros(1,Qnum);
R2_rep_te=zeros(1,Qnum);    RMSE_rep_te=zeros(1,Qnum);
for q=1:Qnum
    if n_te_samp>=2
        y  = sample_true_te(:,q); yh = sample_pred_te_mean(:,q);
        SSE=sum((y-yh).^2); SST=sum((y-mean(y)).^2);
        R2_sample_te(q)=1-SSE/SST; RMSE_sample_te(q)=sqrt(mean((y-yh).^2));
    else
        R2_sample_te(q)=NaN; RMSE_sample_te(q)=NaN;
    end
    y_rep  = Y_rep(is_ext,q); yh_rep = Yhat_test_rep(:,q);
    SSEr=sum((y_rep-yh_rep).^2); SSt=sum((y_rep-mean(y_rep)).^2);
    R2_rep_te(q)=1-SSEr/SSt; RMSE_rep_te(q)=sqrt(mean((y_rep-yh_rep).^2));
end

%% ===== 展示：外层每折 bestLV + 最终 LV =====
fig1=figure('Name','NestedCV_bestLV_per_outer_fold','Position',[60 60 620 420]);
bar(valid_best); grid on; xlabel('外层折（训练池）'); ylabel('best LV');
title(sprintf('嵌套CV：各外层折的最佳LV（outerK=%d, innerK=%d） | final LV=%d', outerK_eff, innerK, bestLV_final));
saveas(fig1, fullfile(out_dir,'nestedCV_bestLV_bar.png'));

%% ===== “像截图那样”的误差棒图（训练嵌套CV + 外测） =====
for q=1:Qnum
    fig=figure('Name',['Err_nestedCV_with_EXT_',response_names{q}], 'Position',[640 80 640 500]);
    hold on; grid on; box on;

    % 训练池：样品级折外（均值±STD）— 蓝色圆点
    sel = ismember(1:N, train_pool_samples)';
    hTrain = errorbar(sample_true_tr(sel,q), sample_pred_tr_mean(sel,q), sample_pred_tr_std(sel,q), ...
        'o', 'LineWidth',1.4, 'MarkerSize',6, 'CapSize',6, 'Color',[0 0.447 0.741]);

    % 外测：样品级（均值±STD）— 红色方块
    hTest = [];
    if n_te_samp>=1
        hTest = errorbar(sample_true_te(:,q), sample_pred_te_mean(:,q), sample_pred_te_std(:,q), ...
            's', 'Color',[0.85 0.1 0.1], 'MarkerFaceColor','none', ...
            'MarkerSize',8, 'LineWidth',1.6, 'CapSize',6, 'LineStyle','none');
    end

    % 1:1 参考线（范围自适应，含两侧）
    vals = [sample_true_tr(sel,q); sample_pred_tr_mean(sel,q)-sample_pred_tr_std(sel,q); ...
            sample_pred_tr_mean(sel,q)+sample_pred_tr_std(sel,q)];
    if n_te_samp>=1
        vals = [vals; sample_true_te(:,q); sample_pred_te_mean(:,q)-sample_pred_te_std(:,q); ...
                      sample_pred_te_mean(:,q)+sample_pred_te_std(:,q)];
    end
    xymin = min(vals); xymax = max(vals); pad = 0.02*(xymax-xymin + eps);
    x1 = xymin - pad; x2 = xymax + pad; h11 = plot([x1 x2],[x1 x2],'k--');

    % 标题与图例
    ttl = sprintf('嵌套CV(train) vs 外测(test) | %s | R^2_t=%.4f, RMSE_t=%.4g | R^2_e=%.4f, RMSE_e=%.4g', ...
        response_names{q}, R2_sample_tr(q), RMSE_sample_tr(q), R2_sample_te(q), RMSE_sample_te(q));
    title(ttl); xlabel(['真实 ',response_names{q}]); ylabel(['预测 ',response_names{q},'（均值±STD）']);
    if ~isempty(hTest)
        legend([hTrain h11 hTest], {'train (nestedCV) mean±STD','1:1 line','external test mean±STD'}, 'Location','best');
    else
        legend([hTrain h11], {'train (nestedCV) mean±STD','1:1 line'}, 'Location','best');
    end
    set(gca,'Layer','top'); axis([x1 x2 x1 x2]); hold off;
    saveas(fig, fullfile(out_dir, sprintf('errorbar_nestedCV_with_ext_%s.png', response_names{q})));
end

%% ===== （可选）外测细节图：灰色复测散点 + 红色方块±STD =====
for q = 1:Qnum
    if n_te_samp < 1, continue; end
    x_true_te = sample_true_te(:,q);
    y_mean_te = sample_pred_te_mean(:,q);
    y_std_te  = sample_pred_te_std(:,q);

    fig = figure('Name',['Err_test_all_',response_names{q}], 'Position',[640 520 640 500]);
    hold on; grid on; box on;

    % 复测级散点（灰色，半透明）
    hRep = [];
    for ii = 1:n_te_samp
        t   = ext_samples_sorted(ii);
        idx = (G_rep(find(is_ext)) == t); %#ok<FNDSB>
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
    x1 = xymin - pad; x2 = xymax + pad; h11 = plot([x1 x2],[x1 x2],'k--');

    % 样品编号标注
    for ii = 1:n_te_samp
        text(x_true_te(ii), y_mean_te(ii), sprintf('  #%d', ext_samples_sorted(ii)), ...
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

%% ===== 新增：VIP 分数（基于最终模型；全谱/ROI 上） =====
% 参考公式：VIP_j = sqrt( p * sum_a( SSY_a * (w_{j,a}^2 / ||w_a||^2) ) / sum_a SSY_a )
% 其中 SSY_a = (t_a' * t_a) * sum_r q_{a,r}^2
A = bestLV_final; pfeat = size(stats_final.W,1);
W = stats_final.W(:,1:A);            % p×A, X-weights
Pload = XL_all(:,1:A);               % p×A, X-loadings（用主输出，避免老版本stats缺P）
% 兼容不同 MATLAB 版本的 Y-loadings 维度（可能是 A×Q 或 Q×A）
if size(YL_all,1)==A && size(YL_all,2)==Qnum
    Qload = YL_all;              % A×Q
elseif size(YL_all,2)==A && size(YL_all,1)==Qnum
    Qload = YL_all.';            % Q×A → A×Q
else
    error('Unexpected size of YL_all: %dx%d (expect A×Q or Q×A).', size(YL_all,1), size(YL_all,2));
end
Ttr = XS_all(:,1:A);                 % n_train×A（训练池的标准化得分）
SSYcomp = sum(Qload.^2,2)'.*sum(Ttr.^2,1);   % 1×A
normW2  = sum(W.^2,1);                        % 1×A
VIP = zeros(pfeat,1);
for j = 1:pfeat
    VIP(j) = sqrt( pfeat * sum( SSYcomp .* ( (W(j,:).^2) ./ max(normW2,eps) ) ) / max(sum(SSYcomp),eps) );
end
% 导出与绘图（VIP 随波长；阈值=1）
vip_tbl = table(wl_roi, VIP, 'VariableNames',{'wavelength_nm','VIP'});
writetable(vip_tbl, fullfile(out_dir,'vip_fullspec_final.csv'));
fig_vip = figure('Name','VIP_full_spectrum','Position',[80 80 820 360]);
plot(wl_roi, VIP, '-'); hold on; yline(1,'r--','VIP=1','LabelHorizontalAlignment','left'); grid on
xlabel('Wavelength (nm)'); ylabel('VIP'); title(sprintf('VIP（final LV=%d）', A));
saveas(fig_vip, fullfile(out_dir,'vip_curve.png'));

%% ===== 新增：逐指标 VIP（每个目标单独的 VIP 光谱与 CSV） =====
colnames = matlab.lang.makeValidName("VIP_"+string(response_names));
VIP_by_target = zeros(pfeat, Qnum);
for qidx = 1:Qnum
    SSYcomp_q = (Qload(:,qidx)'.^2).*sum(Ttr.^2,1);   % 1×A，仅用该目标的Q-loadings
    VIP_q = zeros(pfeat,1);
    for j = 1:pfeat
        VIP_q(j) = sqrt( pfeat * sum( SSYcomp_q .* ((W(j,:).^2)./max(normW2,eps)) ) / max(sum(SSYcomp_q),eps) );
    end
    VIP_by_target(:,qidx) = VIP_q;
    % 单目标 CSV + 曲线
    vip_tbl_q = table(wl_roi, VIP_q, 'VariableNames',{'wavelength_nm','VIP'});
    writetable(vip_tbl_q, fullfile(out_dir, sprintf('vip_fullspec_final_%s.csv', response_names{qidx})));
    fig_vip_q = figure('Name', ['VIP_',response_names{qidx}], 'Position',[100 120 820 360]);
    plot(wl_roi, VIP_q, '-'); hold on; yline(1,'r--','VIP=1','LabelHorizontalAlignment','left'); grid on
    xlabel('Wavelength (nm)'); ylabel('VIP'); title(sprintf('VIP — %s（LV=%d）', response_names{qidx}, A));
    saveas(fig_vip_q, fullfile(out_dir, sprintf('vip_curve_%s.png', response_names{qidx})));
end
% 汇总表（宽格式）
vip_all_tbl = [ table(wl_roi, 'VariableNames', {'wavelength_nm'}), array2table(VIP_by_target, 'VariableNames', colnames) ];
writetable(vip_all_tbl, fullfile(out_dir,'vip_fullspec_final_allTargets_wide.csv'));

%% ===== 新增：Hotelling T^2 与 Q 残差 —— 训练(池) vs 外测 =====
% 投影矩阵 R，用于将新样本投到训练得分空间
Rproj = W / (Pload' * W + eps*eye(A));     % p×A
T_train = Xz_all * Rproj;                  % 训练池得分（与 XS_all 接近）
T_test  = Xz_te  * Rproj;                  % 外测得分
% Hotelling T^2（按每个分量的训练方差标准化）
varTa  = var(T_train, 0, 1); varTa(varTa==0)=eps;
T2_tr  = sum( (T_train.^2) ./ (ones(size(T_train,1),1)*varTa), 2 );
T2_te  = sum( (T_test.^2)  ./ (ones(size(T_test,1),1) *varTa),  2 );
% Q 残差（SPE）：Xz - T*P'
Xhat_tr = T_train * Pload'; E_tr = Xz_all - Xhat_tr; Q_tr = sum(E_tr.^2,2);
Xhat_te = T_test  * Pload'; E_te = Xz_te  - Xhat_te; Q_te = sum(E_te.^2,2);
% 经验控制线（95% 分位）
if exist('prctile','file')
    T2_thr = prctile(T2_tr, 95); Q_thr = prctile(Q_tr, 95);
else
    T2_thr = quantile(T2_tr, 0.95); Q_thr = quantile(Q_tr, 0.95);
end

fig_t2q = figure('Name','T2_vs_Q_train_vs_test','Position',[120 120 700 520]);
loglog(T2_tr, Q_tr, 'o', 'MarkerSize',5, 'Color',[0 0.447 0.741], 'MarkerFaceColor','none'); hold on;
loglog(T2_te, Q_te, 's', 'MarkerSize',6, 'Color',[0.85 0.1 0.1], 'MarkerFaceColor','none');
% 阈值线
xline(T2_thr,'k--','T^2 95%','LabelOrientation','horizontal','LabelVerticalAlignment','bottom');
yline(Q_thr, 'k--','Q 95%','LabelHorizontalAlignment','left');
grid on; xlabel('Hotelling T^2 (log)'); ylabel('Q residual / SPE (log)');
title(sprintf('T^2 - Q（train pool vs external） | LV=%d', A));
legend({'train pool','external test'}, 'Location','best'); hold off;
saveas(fig_t2q, fullfile(out_dir,'t2_q_scatter_train_vs_external.png'));

% 导出 T^2 / Q（方便排查离群）
T2Q_train = table(find(is_pool), T2_tr, Q_tr, 'VariableNames',{'row_idx_in_pool','T2','Q'});
writetable(T2Q_train, fullfile(out_dir,'t2q_train_pool.csv'));
T2Q_test  = table(find(is_ext),  T2_te, Q_te, 'VariableNames',{'row_idx_in_external','T2','Q'});
writetable(T2Q_test,  fullfile(out_dir,'t2q_external.csv'));

%% ===== 导出结果 =====
% 复测级：训练池的嵌套CV折外 + 外测一次性预测
T_rep = table(G_rep, R_idx, 'VariableNames',{'sample_idx','rep_idx'});
for q=1:Qnum
    T_rep.(['true_',response_names{q}]) = Y_rep(:,q);
    col_tr = nan(n_obs,1); col_tr(is_pool) = yhat_outercv(is_pool,q);
    col_te = nan(n_obs,1); col_te(is_ext)  = Yhat_test_rep(:,q);
    T_rep.(['pred_nestedCV_train_',response_names{q}]) = col_tr;
    T_rep.(['pred_external_',response_names{q}])       = col_te;
end
writetable(T_rep, fullfile(out_dir,'predictions_replicate_level_trainNestedCV_and_external.csv'));

% 样品级：训练池（嵌套CV）
tr_idx_sorted = sort(train_pool_samples);
T_samp_tr = table(tr_idx_sorted(:), sample_nrep_tr(tr_idx_sorted), 'VariableNames',{'sample_idx','n_reps'});
for q=1:Qnum
    T_samp_tr.(['true_',response_names{q}])     = sample_true_tr(tr_idx_sorted,q);
    T_samp_tr.(['predMean_',response_names{q}]) = sample_pred_tr_mean(tr_idx_sorted,q);
    T_samp_tr.(['predSTD_',response_names{q}])  = sample_pred_tr_std(tr_idx_sorted,q);
end
writetable(T_samp_tr, fullfile(out_dir,'predictions_sample_level_train_nestedCV.csv'));

% 样品级：外测
T_samp_te = table(ext_samples_sorted(:), sample_nrep_te, 'VariableNames',{'sample_idx','n_reps'});
for q=1:Qnum
    T_samp_te.(['true_',response_names{q}])     = sample_true_te(:,q);
    T_samp_te.(['predMean_',response_names{q}]) = sample_pred_te_mean(:,q);
    T_samp_te.(['predSTD_',response_names{q}])  = sample_pred_te_std(:,q);
end
writetable(T_samp_te, fullfile(out_dir,'predictions_sample_level_external.csv'));

% 指标汇总
T_metrics = table(response_names(:), repmat(bestLV_final,Qnum,1), ...
    R2_sample_tr(:), RMSE_sample_tr(:), R2_rep_tr(:), RMSE_rep_tr(:), ...
    R2_sample_te(:), RMSE_sample_te(:), R2_rep_te(:), RMSE_rep_te(:), ...
'VariableNames',{'target','bestLV_final','R2_sample_train_nestedCV','RMSE_sample_train_nestedCV','R2_rep_train_nestedCV','RMSE_rep_train_nestedCV', ...
                 'R2_sample_external','RMSE_sample_external','R2_rep_external','RMSE_rep_external'});
writetable(T_metrics, fullfile(out_dir,'metrics_trainNestedCV_and_external.csv'));

% 保存最终模型（仅训练池拟合）
model=struct();
model.betaZ=beta_all; model.nLV=bestLV_final; model.zscore_X=zscore_X; model.zscore_Y=zscore_Y;
model.muX=muX_all; model.sigX=sigX_all; model.muY=muY_all; model.sigY=sigY_all;
model.wavelength=wl_roi; model.sample_names=spec_names(:); model.response_names=string(response_names(:));
model.use_lines_only=use_lines_only; model.lines_nm=lines_nm(:); model.win_nm=win_nm;
model.outerK=outerK_eff; model.innerK=innerK_eff; model.bestLV_outer=bestLV_outer; model.bestLV_final=bestLV_final;
model.train_pool_idx = sort(train_pool_samples(:))'; model.external_idx = ext_samples_sorted(:)';
model.random_seed = rng_seed;
save(fullfile(out_dir,'plsr_model_trainPoolOnly_withExternal.mat'),'-struct','model','-v7.3');

fprintf('\n[DONE] 嵌套CV（训练池）+ 外测保留评估 + VIP + T^2/Q 已完成。输出目录：%s\n', out_dir);
