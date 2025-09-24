%{
20230411-DN

%}
function [mat_av,mat_std,noise_mat,SNR_av,SNR_std,noise_mat_SBR,SBR_av,SBR_std] = Calculate2(wave,dataset,wave_slec,SeriesNum,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'dataset',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'label',validScalarPosNum);
    addOptional(p,'subbase',defaultcoe);
    parse(p,wave,dataset,wave_slec,SeriesNum,varargin{:});
    
    % 判断wave中是否自带噪声范围，只有一列则不带噪声，需计算
    noisecal_flag = 0;
    if size(wave_slec,2) == 1
        noisecal_flag = 1;
    end
    subbase_flag = 0;
    if p.Results.subbase == 1
        subbase_flag = 1;
    end
   %% 数据集 读取
    sample_num = round(size(dataset,1)/SeriesNum);   %共有n个种类
    mat_av = [];    %平均值
    mat_std = [];   %std
    SNR_av = [];    %平均值
    SNR_std = [];   %std
    SBR_av = [];    %平均值
    SBR_std = [];   %std
    %% 噪声范围计算*******************************************************
    if noisecal_flag == 0 %自带噪声范围
        noise_mat = zeros(SeriesNum,size(wave_slec,1));
        noise_mat_SBR = zeros(SeriesNum,size(wave_slec,1));
        for i = 1:size(wave_slec,1)
            noise_class = [];
            noise_class_SBR = [];
            for j = 1:SeriesNum
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                [~,index_1] = min(abs(wave_slec(i,2)-wave));
                [~,index_2] = min(abs(wave_slec(i,3)-wave));
                noise_fragment = line(index_1:index_2);
                noise_SBR = mean(noise_fragment);
                noise = std(noise_fragment);
                noise_class = cat(2,noise_class,noise);   %按列合并
                noise_class_SBR = cat(2,noise_class_SBR,noise_SBR);   %按列合并
            end
            noise_mat(:,i) = noise_class';
            noise_mat_SBR(:,i) = noise_class_SBR';
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
    
%% 强度、信噪比计算
    for i = 1:SeriesNum
        dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
        % 计算均值 方差
%         [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
        [line,~] = findpeak_7points(wave_slec(:,1),wave,dataset_t);
        % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
        % 第三个参数代表的是按照列求标准差还是按照行求标准差
        st = std(line,1,1);
        SNR = line./noise_mat(i,:);
        SBR = line./noise_mat_SBR(i,:);
        if subbase_flag == 0 %不需要去基线 
            line = mean(dataset_t,1);
%             [av,~] = manualline(wave_slec(:,1),wave,line);
            [av,~] = findpeak_7points(wave_slec(:,1),wave,line);
        elseif subbase_flag == 1 % 去基线 
            line = mean(dataset_t,1);
            [subline, ~] = SubBackground_ModelFree(line, 30);
%             [subline,~] =  baseline_arPLS(line);
%             [av,~] = manualline(wave_slec(:,1),wave,subline);
            [av,~] = findpeak_7points(wave_slec(:,1),wave,subline);
        end
        av_snr = mean(SNR,1);
        std_snr = std(SNR,1,1);
        av_sbr = mean(SBR,1);
        std_sbr = std(SBR,1,1);
%       st = std(line,1,1);
        SNR_av = cat(1,SNR_av,av_snr);
        SNR_std = cat(1,SNR_std,std_snr);
        SBR_av = cat(1,SBR_av,av_sbr);
        SBR_std = cat(1,SBR_std,std_sbr);
        mat_av = cat(1,mat_av,av);
        mat_std = cat(1,mat_std,st);
    end 
end
