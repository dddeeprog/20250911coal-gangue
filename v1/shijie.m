%% 修订版：跳过 xlsx 定标 + **特定路径读取**（CSV 首列=波长，其余列=强度）+ **全谱定量** + **限定文件名顺序（1,2,3,4,5,6）**
% 依赖：
%   - Find_csv2.m （首列=波长，其余列=强度；对齐公共波长轴；支持文件名清单）
%   - Spectrum_allneed.m, LinearFit.m, PolyfitCV.m, CalDensityNum2.m,
%     TestPerform3.m, PolyfitCV3.m
%
% 说明：
%   - 训练/测试的公共波长轴由“读取到的第一批 CSV”自动确定；
%   - 所有外部文件/目录采用 **固定绝对路径**，请在“配置区”按你的实际路径修改；
%   - 在原“多谱线窗口 + PLSR”的基础上，新增 **全谱定量（Full-spectrum PLSR）**；
%   - 新增 **文件名白名单**：可仅按 {'1','2','3','4','5','6'} 的顺序读取 CSV。

clear; clc; close all

%% ===== 1) 配置区（按需修改为你的实际路径） =====
% 数据根目录（包含 01-10-<档名>\1 和可选的 \2）
DATA_ROOT = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B-test';

% 标签（热值/C 含量等）与谱线清单 Excel 的绝对路径
PATH_DENSITY_XLSX = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B-test\targets\HotC.xlsx';
PATH_LINES_XLSX   = 'C:\Users\TomatoK\Desktop\20250911coal&gangue\gangue\spec\ave_5\110A110B-test\targets\HotC.xlsx';

% 档位名称（顺序要与标签 Excel 的行顺序一致）
namee  = {'1','2','3','4','5','6'};  % 共 7 档
num    = 6;        % 档位数量
numone = 50;       % 每档最多取的样本数（按“强度列”计）
MeanNum = 5;       % 每 MeanNum 条做一次平均

% 仅按给定基名顺序读取 CSV（不带扩展名）。例如 1,2,3,4,5,6
USE_FILE_LIST = true;
FILE_BASES = arrayfun(@num2str, 1:6, 'uni', false);  % {'1','2','3','4','5','6'}

% 单线定标参数（示例）
Range       = 2;            % ±2 nm 窗口
ElementLine = 247.909;      % 目标谱线（示例）

% 全谱定量的设置
FULLSPEC_ENABLE   = true;    % 是否执行全谱 PLSR
FULLSPEC_PREPROC  = 'none';% 预处理：'none' 或 'zscore'（按列标准化）
FULLSPEC_MAX_NC   = 30;      % 扫描的最大潜变量数上限（会被样本数-1限制）

% 基本存在性检查
assert(isfolder(DATA_ROOT), '数据根目录不存在: %s', DATA_ROOT);
assert(isfile(PATH_DENSITY_XLSX), '未找到标签文件: %s', PATH_DENSITY_XLSX);
assert(isfile(PATH_LINES_XLSX) || ~exist('PATH_LINES_XLSX','var'), '未找到谱线清单: %s', PATH_LINES_XLSX);

%% ===== 2) 读取训练集（CSV 首列=波长，其余列=强度），并确定公共波长轴 =====
SpecTrain        = [];
num_sum          = [0];               % 统计每档样本数（用于后续分块求均值/方差）
MergeWavelenRef  = [];                % 统一的公共波长轴（由第一个成功读取的 CSV 决定）

for i = 1:num
    selpath = fullfile(DATA_ROOT, ['01-10-' namee{i}], '1');
    if ~isfolder(selpath)
        error('训练集子目录不存在: %s', selpath);
    end
    if USE_FILE_LIST
        [Spec1, MergeWavelenRef] = Find_csv2(selpath, numone, MergeWavelenRef, FILE_BASES);
    else
        [Spec1, MergeWavelenRef] = Find_csv2(selpath, numone, MergeWavelenRef);
    end
    SpecTrain = [SpecTrain, Spec1];
    num_sum   = [num_sum, size(Spec1,2)];
end
MergeWavelen = MergeWavelenRef;  % 传给后续窗口函数

%% ===== 3) 读取测试集（优先使用 \\2；若无则回退到 \\1） =====
SpecTest = [];
for i = 1:num
    pref2   = fullfile(DATA_ROOT, ['01-10-' namee{i}], '2');
    selpath = pref2;
    if ~isfolder(pref2)
        selpath = fullfile(DATA_ROOT, ['01-10-' namee{i}], '1');
        warning('测试集未找到 %s ，回退到同批次 \\1。', pref2);
    end
    if USE_FILE_LIST
        [Spec1, ~] = Find_csv2(selpath, numone, MergeWavelenRef, FILE_BASES);
    else
        [Spec1, ~] = Find_csv2(selpath, numone, MergeWavelenRef);
    end
    SpecTest = [SpecTest, Spec1];
end

%% ===== 4) 每 MeanNum 条做一次平均（降噪） =====
assert(mod(size(SpecTrain,2), MeanNum) == 0, '训练集列数不能被 MeanNum 整除');
assert(mod(size(SpecTest, 2), MeanNum) == 0, '测试集列数不能被 MeanNum 整除');

MeanSpecTrain = zeros(size(SpecTrain,1), size(SpecTrain,2)/MeanNum);
MeanSpecTest  = zeros(size(SpecTest,1),  size(SpecTest, 2)/MeanNum);
for i = 1 : size(MeanSpecTrain,2)
    MeanSpecTrain(:,i) = mean(SpecTrain(:, (i-1)*MeanNum + (1:MeanNum)), 2);
    MeanSpecTest(:,i)  = mean(SpecTest(:,  (i-1)*MeanNum + (1:MeanNum)), 2);
end
num_sum2 = num_sum/MeanNum;   % 各档平均后的样本数
aaa = cumsum(num_sum2);       % 分界索引

%% ===== 5) 单谱线定标（关注 RSD） =====
DensityAll = readmatrix(PATH_DENSITY_XLSX); % 8-热值  9-C （当前取第 8 列）
density    = DensityAll(:,8);

[~, SpecLin_mean, ~] = Spectrum_allneed(ElementLine, MeanSpecTrain, Range, MergeWavelen);

result      = zeros(num,3);
result(:,1) = density(:);
for i = 1:num
    result(i,2) = mean(SpecLin_mean(aaa(i)+1 : aaa(i+1)));
    result(i,3) = std( SpecLin_mean(aaa(i)+1 : aaa(i+1)) );
end

[Slope,Intercept,R2,Pearson_r,Adj_R_Square] = LinearFit(result(:,1), result(:,2), result(:,3)); %#ok<ASGLU>
resultA = (result(:,2)-Intercept)/Slope;  %#ok<NASGU>

RSD = mean(result(:,3)./result(:,2));
[~, RMSECV, ARE] = PolyfitCV(result(:,1), result(:,2), 5);
pass1 = [R2; RSD; RMSECV; ARE]; %#ok<NASGU>

figure; scatter(result(:,1), result(:,2), 36, 'filled'); xlabel('标签'); ylabel('窗口强度'); title('单线定标散点'); grid on

%% ===== 6) 构造逐样本标签（与平均后样本数匹配） =====
rep_per_group = numone/MeanNum;  % 50/5=10
TrainDensityAll = repelem(density(:), rep_per_group);
TestDensityAll  = repelem(density(:), rep_per_group);

%% ===== 7) 多谱线窗口特征 + PLSR（原有功能） =====
if isfile(PATH_LINES_XLSX)
    Lineneed = readmatrix(PATH_LINES_XLSX); 
    Linee    = Lineneed(:,1);
    [~, TrainSpec, ~] = Spectrum_allneed(Linee, MeanSpecTrain, Range, MergeWavelen);  % 样本×特征
    [~, TestSpec,  ~] = Spectrum_allneed(Linee, MeanSpecTest,  Range, MergeWavelen);

    DTrainHOGHuge = TrainSpec;  % 与原命名保持一致
    DTestHOGHuge  = TestSpec;

    TResult = [];
    kk = 0;
    for ncomp_need = 2:2:26
        kk = kk + 1;
        [XL,YL,XS,YS,BETAHuge,PCTVAR,MSE,stats] = plsregress(DTrainHOGHuge, TrainDensityAll, ncomp_need, 'cv', 5); %#ok<ASGLU>
        
        % 训练集预测
        TrainHugeplsr   = [ones(size(DTrainHOGHuge,1),1), DTrainHOGHuge] * BETAHuge;                
        TrainHugeresult = TrainHugeplsr(:,1);

        % 按档聚合
        [Tnum, AllPar] = CalDensityNum2(TrainHugeresult, TrainDensityAll);

        resultTrainHuge      = zeros(num,3);
        resultTrainHuge(:,1) = Tnum(2,2:end)';
        aaaTrainHuge         = cumsum([Tnum(1,:)]);
        for ii = 1:numel(aaaTrainHuge)-1
            resultTrainHuge(ii,2) = mean(AllPar(aaaTrainHuge(ii)+1 : aaaTrainHuge(ii+1), 1));
            resultTrainHuge(ii,3) = std( AllPar(aaaTrainHuge(ii)+1 : aaaTrainHuge(ii+1), 1));
        end

        [Slope,Intercept,R2,~,~] = LinearFit(resultTrainHuge(:,1), resultTrainHuge(:,2), resultTrainHuge(:,3)); %#ok<ASGLU>
        [~, RMSECV, ARECV]       = PolyfitCV(resultTrainHuge(:,1), resultTrainHuge(:,2), 5);
        RSD = mean(resultTrainHuge(:,3)./resultTrainHuge(:,2));

        TResult(1,kk) = 1;
        TResult(2,kk) = 1;
        TResult(3,kk) = ncomp_need;
        TResult(4,kk) = R2;
        TResult(5,kk) = ARECV;
        TResult(6,kk) = RMSECV;
        TResult(7,kk) = RSD;

        % 测试集
        TestHugeplsr   = [ones(size(DTestHOGHuge,1),1), DTestHOGHuge] * BETAHuge;                  
        TestHugeresult = TestHugeplsr(:,1);

        [Tnum, AllPar] = CalDensityNum2(TestHugeresult, TestDensityAll);

        % 用训练端线性映射在测试端评估
        resultTestHuge      = zeros(num,3);
        resultTestHuge(:,1) = Tnum(2,2:end)';
        aaaTestHuge         = cumsum([Tnum(1,:)]);
        for ii = 1:numel(aaaTestHuge)-1
            resultTestHuge(ii,2) = mean(AllPar(aaaTestHuge(ii)+1 : aaaTestHuge(ii+1), 1));
            resultTestHuge(ii,3) = std( AllPar(aaaTestHuge(ii)+1 : aaaTestHuge(ii+1), 1));
        end

        R2Test        = TestPerform3(Slope, Intercept, resultTestHuge);
        [RMSEp, AREp] = PolyfitCV3(resultTestHuge(:,1), resultTestHuge(:,2), Slope, Intercept);

        RSD = mean(resultTestHuge(:,3)./resultTestHuge(:,2));
        TResult(8,kk)  = R2Test;
        TResult(9,kk)  = AREp;
        TResult(10,kk) = RMSEp;
        TResult(11,kk) = RSD;
    end
else
    warning('未提供谱线清单文件，已跳过窗口特征 PLSR。');
end

%% ===== 7b) 全谱定量（Full-spectrum PLSR） =====
if FULLSPEC_ENABLE
    % 构造“样本 × 特征”的全谱矩阵（以平均后的光谱为基础）
    % MeanSpecTrain/Test 当前是 [N_wav × N_sample]，转置后为 [N_sample × N_wav]
    Xtr_full = (MeanSpecTrain)';
    Xte_full = (MeanSpecTest)';

    % 去除零方差列（防止标准化除以 0）
    varX = var(Xtr_full, 0, 1);
    keep = varX > 0;
    Xtr_full = Xtr_full(:, keep);
    Xte_full = Xte_full(:, keep);

    % 预处理
    switch lower(FULLSPEC_PREPROC)
        case 'zscore'
            [Xtr_full, muX, sigX] = zscore(Xtr_full, 0, 1);
            sigX(sigX==0) = 1; % 保险
            Xte_full = (Xte_full - muX)./sigX;
        case 'none'
            % 不做处理
        otherwise
            error('未知 FULLSPEC_PREPROC: %s', FULLSPEC_PREPROC);
    end

    % 成分数网格（受样本数限制）
    nmax = min(FULLSPEC_MAX_NC, size(Xtr_full,1)-1);
    if nmax < 2
        warning('全谱：可用样本过少，跳过 PLSR。');
    else
        ncomp_grid_full = 2:2:nmax;
        TResult_full = zeros(11, numel(ncomp_grid_full));
        kk = 0;
        for ncomp_need = ncomp_grid_full
            kk = kk + 1;
            [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(Xtr_full, TrainDensityAll, ncomp_need, 'cv', 5); %#ok<ASGLU>

            % 训练集预测（逐样本）
            ytr_hat = [ones(size(Xtr_full,1),1), Xtr_full]*BETA;

            % 档内聚合
            [Tnum, AllPar] = CalDensityNum2(ytr_hat, TrainDensityAll);
            resultTrain = zeros(num,3);
            resultTrain(:,1) = Tnum(2,2:end)';
            aaaTrain = cumsum([Tnum(1,:)]);
            for ii = 1:numel(aaaTrain)-1
                resultTrain(ii,2) = mean(AllPar(aaaTrain(ii)+1 : aaaTrain(ii+1), 1));
                resultTrain(ii,3) = std( AllPar(aaaTrain(ii)+1 : aaaTrain(ii+1), 1));
            end

            [SlopeF,InterceptF,R2F,~,~] = LinearFit(resultTrain(:,1), resultTrain(:,2), resultTrain(:,3)); %#ok<ASGLU>
            [~, RMSECVF, AREF]         = PolyfitCV(resultTrain(:,1), resultTrain(:,2), 5);
            RSDF = mean(resultTrain(:,3)./resultTrain(:,2));

            TResult_full(1,kk) = 1;
            TResult_full(2,kk) = 1;
            TResult_full(3,kk) = ncomp_need;
            TResult_full(4,kk) = R2F;
            TResult_full(5,kk) = AREF;
            TResult_full(6,kk) = RMSECVF;
            TResult_full(7,kk) = RSDF;

            % 测试集预测
            yte_hat = [ones(size(Xte_full,1),1), Xte_full]*BETA;
            [Tnum, AllPar] = CalDensityNum2(yte_hat, TestDensityAll);
            resultTest = zeros(num,3);
            resultTest(:,1) = Tnum(2,2:end)';
            aaaTest = cumsum([Tnum(1,:)]);
            for ii = 1:numel(aaaTest)-1
                resultTest(ii,2) = mean(AllPar(aaaTest(ii)+1 : aaaTest(ii+1), 1));
                resultTest(ii,3) = std( AllPar(aaaTest(ii)+1 : aaaTest(ii+1), 1));
            end

            R2TestF        = TestPerform3(SlopeF, InterceptF, resultTest);
            [RMSEpF, AREpF] = PolyfitCV3(resultTest(:,1), resultTest(:,2), SlopeF, InterceptF);
            RSDF_test      = mean(resultTest(:,3)./resultTest(:,2));

            TResult_full(8,kk)  = R2TestF;
            TResult_full(9,kk)  = AREpF;
            TResult_full(10,kk) = RMSEpF;
            TResult_full(11,kk) = RSDF_test;
        end

        % 可选：快速对比图（测试端 RMSE）
        figure; plot(ncomp_grid_full, TResult_full(10,:), '-o'); grid on
        xlabel('潜变量个数'); ylabel('RMSE_p（测试端）'); title('全谱 PLSR：成分数-RMSE_p');
    end
end

% 输出（可选）
% if exist('TResult','var'),       writematrix(TResult,       'plsr_metrics_window.csv'); end
% if exist('TResult_full','var'),  writematrix(TResult_full,  'plsr_metrics_fullspec.csv'); end
% save('plsr_workspace.mat');

disp('Done. 已使用固定路径读取数据，按 1..6 的文件顺序（可配置）读取 CSV，并执行窗口特征与全谱 PLSR 定量。')