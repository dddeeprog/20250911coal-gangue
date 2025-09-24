function [Yhat, T, info] = predict_from_csv_single(model_mat_path, csv_path, varargin)
%PREDICT_FROM_CSV_SINGLE  用已保存的 PLSR 模型预测“单幅 CSV 光谱”（免外部依赖）。
%
% [Yhat, T, info] = predict_from_csv_single(model_mat_path, csv_path, ...)
%
% 输入
%   model_mat_path : 训练脚本导出的模型 .mat（如 plsr_model_multi_xlsx.mat / _PLS1.mat）
%   csv_path       : 单幅光谱的 CSV 文件路径
%
% 名值对参数（可选）
%   'WavelengthColumn' : 波长所在列（默认 1）
%   'IntensityColumn'  : 强度所在列（默认 2）
%   'ExportCSV'        : true/false（默认 false）
%   'OutDir'           : 导出目录（默认与模型同目录）
%
% 说明
%   - 若 CSV 只有一列且长度与模型波长数 P 相同，则视为“仅强度”，波长使用模型自带的 wavelength。
%   - 若 CSV 有两列或更多，默认第1列=波长、第2列=强度；可用参数修改。
%   - 自动识别微米：若 max(wl)∈(0.1, 10)，将视为微米并 ×1000 转为 nm。
%
% 2025-09

%% 解析参数
p = inputParser; p.KeepUnmatched = true;
addParameter(p,'WavelengthColumn',1,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'IntensityColumn',2,@(x)isnumeric(x)&&isscalar(x)&&x>=1);
addParameter(p,'ExportCSV',false,@islogical);
addParameter(p,'OutDir','',@(x)ischar(x)||isstring(x));
parse(p,varargin{:});
Wc = p.Results.WavelengthColumn; Ic = p.Results.IntensityColumn;
doCSV = p.Results.ExportCSV; outdir = string(p.Results.OutDir);

%% 读取模型（拿到模型波长与标准化参数）
m = load(model_mat_path);
if isfield(m,'betaZ')
    beta = m.betaZ;                    % (P+1)×Q
elseif isfield(m,'betaZ_cols')
    beta = m.betaZ_cols;               % (P+1)×Q（旧命名）
else
    error('模型缺少 betaZ/betaZ_cols 字段。');
end
req = {'wavelength','muX','sigX','muY','sigY'};
for k=1:numel(req)
    if ~isfield(m,req{k}), error('模型缺少字段：%s', req{k}); end
end
wl_model = m.wavelength(:);  P = numel(wl_model);
muX = m.muX(:)'; sigX = m.sigX(:)'; muY = m.muY(:)'; sigY = m.sigY(:)';

% 指标名
if isfield(m,'response_names')
    rn = string(m.response_names);
else
    rn = "y"+string(1:size(beta,2));
end
varnames = cellstr(matlab.lang.makeValidName(rn));

%% 读取 CSV（读成数值矩阵）
M = readmatrix(csv_path);
if isempty(M) || ~isnumeric(M), error('无法从 CSV 读取数值内容：%s', csv_path); end
if size(M,1)==1 && size(M,2)>1, M = M.'; end  % 行向量转列

%% 解析 wl 与强度
unit_hint = 'nm';
if size(M,2) >= 2
    if Wc>size(M,2) || Ic>size(M,2)
        error('列索引超出范围：CSV 只有 %d 列。', size(M,2));
    end
    wl = M(:,Wc); y = M(:,Ic);
    ok = isfinite(wl) & isfinite(y);
    wl = wl(ok); y = y(ok);
    if max(wl) < 10 && max(wl) > 0.1   % 微米 → 纳米
        wl = wl * 1000; unit_hint = 'um->nm';
    end
    if ~issorted(wl), [wl,ord] = sort(wl); y = y(ord); end
    wl_new = wl(:); spec_new = y(:);   % M×1
else
    % 单列：视为仅强度，长度需等于模型 P
    if numel(M) ~= P
        error(['CSV 仅 1 列/向量，但长度(%d)≠模型波长数 P(%d)。', ...
               ' 请提供两列（波长,强度）或导出与模型同网格的强度向量。'], numel(M), P);
    end
    wl_new = wl_model; spec_new = M(:);
    unit_hint = 'use_model_nm';
end

%% —— 调用本文件内置的预测子函数（免外部依赖）——
[Yhat, T, info] = predict_plsr_model_local(m, wl_new, spec_new);

% 附加元数据
info.csv.path = char(csv_path);
info.csv.unit_hint = unit_hint;
info.csv.has_wavelength = (size(M,2) >= 2);
info.model.P = P;

%% 可选导出 CSV
if doCSV
    if strlength(outdir)==0, outdir = string(fileparts(model_mat_path)); end
    if ~exist(outdir,'dir'), mkdir(outdir); end
    [~,base,~] = fileparts(csv_path);
    out_rep = fullfile(outdir, sprintf('predictions_%s_replicate.csv', base));
    writetable(T.rep, out_rep);
end

% 友好打印
disp('--- 复测级预测（单条光谱）---'); disp(T.rep);

end  % ====== 主函数结束 ======

% ===================== 内置预测子函数 =====================
function [Yhat, out_tbl, info] = predict_plsr_model_local(m, wl_new, spec_new)
% 输入：
%   m      : 已 load 的模型结构体（包含 betaZ/betaZ_cols、wavelength、muX/sigX/muY/sigY、response_names）
%   wl_new : 新光谱波长（列向量）
%   spec_new: 新光谱强度（列向量），长度= numel(wl_new)
%
% 输出：
%   Yhat   : 1×Q 复测级预测
%   out_tbl: .rep table（1×Q），列名为 response_names
%   info   : 插值/外推信息等

% 取 beta
if isfield(m,'betaZ')
    beta = m.betaZ;
else
    beta = m.betaZ_cols;
end
wl_model = m.wavelength(:);  P = numel(wl_model); Q = size(beta,2);
muX = m.muX(:)'; sigX = m.sigX(:)'; muY = m.muY(:)'; sigY = m.sigY(:)';

% 对齐波长（插值/外推）
wl_new = wl_new(:); y = spec_new(:);
if numel(wl_new) ~= numel(wl_model) || any(abs(wl_new - wl_model) > 1e-9)
    yi = interp1(wl_new, y, wl_model, 'linear', 'extrap');  % 到模型网格
    lo = sum(wl_model < min(wl_new)); hi = sum(wl_model > max(wl_new));
    info.interp.extrap_left  = lo; info.interp.extrap_right = hi;
    info.interp.extrap_frac  = (lo+hi)/numel(wl_model);
else
    yi = y; info.interp.extrap_frac = 0;
end

% 标准化 + 预测
sigX(sigX==0) = eps;
Xz = (yi(:)' - muX) ./ sigX;    % 1×P
Yhat_z = [1 Xz] * beta;         % 1×Q
Yhat   = Yhat_z .* sigY + muY;  % 1×Q

% 输出表
if isfield(m,'response_names')
    rn = string(m.response_names);
else
    rn = "y"+string(1:Q);
end
varnames = cellstr(matlab.lang.makeValidName(rn));
out_tbl.rep = array2table(Yhat, 'VariableNames', varnames);
info.model = struct('P',P,'Q',Q);
end
