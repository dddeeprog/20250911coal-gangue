%{
20230322-DN
该函数与Calculate不同的地方在于去除了异常值
dataset：所有数据集
wave_slec: 单一元素的多个波长，一行一个波长，用分号分隔 (手动选择峰值点位置更加精确) 
    Cd = [214.473,214,215; 226.560,227,228];手动给定每个元素的噪声范围
    Cd = [214.473;226.560;228.836];自动选择噪声范围
可选参数：
subbase:去除基线后定量 默认不去基线，
    为提升计算速度，先对单类取平均后，全谱去除基线 故只对av起作用 对std LOD(噪声） 无影响
应用实例：
%}
function [mat_av,mat_std,SNR_av,SNR_std] = Paraoptim_outliers(wave,dataset,wave_slec,noise_mat,SeriesNum,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'dataset',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'noise_mat',validScalarPosNum);
    parse(p,wave,dataset,wave_slec,SeriesNum,varargin{:});
    %% 初始化参数
    sample_num = round(size(dataset,1)/SeriesNum);%共有n个种类
    mat_av = [];    %平均值
    mat_std = [];   %std
    SNR_av = [];    %平均值
    SNR_std = [];   %std
    %% 强度信噪比计算*******************************************************
    for i = 1:SeriesNum
        dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
        % 计算均值 方差
        [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
        % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
        % 第三个参数代表的是按照列求标准差还是按照行求标准差
        nan_rows = any(isnan(line),2); % 找到包含NaN的行
        line(nan_rows, :) = [];        % 删除包含NaN的行
        st = std(line,1,1);
        SNR = line./noise_mat(i,:);
        av_snr = mean(SNR,1);
        std_snr = std(SNR,1,1);
        dataset_t(nan_rows, :) = [];
        line = mean(dataset_t,1);
        [av,~] = manualline(wave_slec(:,1),wave,line);
        SNR_av = cat(1,SNR_av,av_snr);
        SNR_std = cat(1,SNR_std,std_snr);
        mat_av = cat(1,mat_av,av);
        mat_std = cat(1,mat_std,st);
    end
end