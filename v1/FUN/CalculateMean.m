%{
20221205-DN
求各个文件夹的平均RSD
datasetall：所有数据集
wave_slec: 单一元素的多个波长，一行一个波长，用分号分隔 (手动选择峰值点位置更加精确) 
    Cd = [214.473,214,215; 226.560,227,228];手动给定每个元素的噪声范围
    Cd = [214.473;226.560;228.836];自动选择噪声范围
label: 元素的浓度和含量
    x = [1,2,4,6,8,10,15,0.1,0.2,0.4,0.6,0.8];
可选参数：
subbase:去除基线后定量 默认不去基线，
    为提升计算速度，先对单类取平均后，全谱去除基线 故只对av起作用 对std LOD(噪声） 无影响
应用实例：
%}
function [rsdmean,rsd_std,SNRMean,SNR_std,IntMean,IntMean_std] = CalculateMean(wave,dataset,wave_slec,SeriesNum,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'dataset',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'label',validScalarPosNum);
    addOptional(p,'subbase',defaultcoe,validScalarPosNum);
    parse(p,wave,dataset,wave_slec,SeriesNum,varargin{:});
    
    % 判断wave中是否自带噪声范围，只有一列则不带噪声，需计算
    noisecal_flag = 0;
    if size(wave_slec,2) == 1
        noisecal_flag = 1;
    end
    % 是否需要去除基线
    subbase_flag = 0;
    if p.Results.subbase ~= 0
        subbase_flag = 1;
    end  
   %% 数据集 读取
    sample_num = round(size(dataset,1)/SeriesNum);%共有n个种类
    mat_av = [];    %平均值
    mat_std = [];   %std
    for i = 1:SeriesNum
        dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
        % 计算均值 方差
        [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
        % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
        % 第三个参数代表的是按照列求标准差还是按照行求标准差
        st = std(line,1,1);
        if subbase_flag == 0 %不需要去基线 
            line = mean(dataset_t,1);
            [av,~] = manualline(wave_slec(:,1),wave,line);
        elseif subbase_flag == 1 % 去基线 
            line = mean(dataset_t,1);
            [subline,~] =  baseline_arPLS(line);
            [av,~] = manualline(wave_slec(:,1),wave,subline);
        end
        mat_av = cat(1,mat_av,av);
        mat_std = cat(1,mat_std,st);
    end
    %% 噪声范围计算*******************************************************
    if noisecal_flag == 0 %自带噪声范围
        noise_mat = zeros(SeriesNum,size(wave_slec,1));
        for i = 1:size(wave_slec,1)
            noise_class = [];
            for j = 1:SeriesNum
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                [~,index_1] = min(abs(wave_slec(i,2)-wave));
                [~,index_2] = min(abs(wave_slec(i,3)-wave));
                noise_fragment = line(index_1:index_2);
                noise = std(noise_fragment);
                noise_class = cat(2,noise_class,noise);   %按列合并
            end
            noise_mat(:,i) = noise_class';
        end
    elseif noisecal_flag == 1% 只有波长一列 根据区域计算噪声范围
        noise_mat = zeros(SeriesNum,size(wave_slec,1));
        for i = 1:size(wave_slec,1)
            wave_t = wave_slec(i);
            [~,index] = min(abs(wave_t-wave));
            noise_class = [];
            for j = 1:SeriesNum
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                noise_fragment = line(max(1,index-20):min(length(line),index+20));
                noise_fragment_sort = sort(noise_fragment);
                noise = std(noise_fragment_sort(1:round(0.8*length(noise_fragment_sort))));
                noise_class = cat(2,noise_class,noise);
            end
            noise_mat(:,i) = noise_class';
        end
    end    
    mat_rsd = mat_std./mat_av;
    rsdmean = mean(mat_rsd,1);
    rsd_std = std(mat_rsd,1,1); 
    SNR = mat_av./noise_mat;
    SNRMean = mean(SNR,1);
    SNR_std = std(SNR,1,1);
    IntMean = mean(mat_av,1);
    IntMean_std = std(mat_av,1,1);
end
