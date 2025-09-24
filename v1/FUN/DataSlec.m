%**************************************************************************
% 代码说明：根据sampleslec feature_slec选择样本和特征 降低数据集
% 输入：dataset wave
%       n 类别数量
%       waveslec：选择的波长
% 输出：
% 数据集：dataset
% 波长：wave
% 文件夹个数：5
% 
% DataSlec(dataset,wave,5,1,wave_slec);
%**************************************************************************
function dataselc = DataSlec(dataset,wave,n,sampleslec,wave_sle)
    samplepoints = round(size(dataset,1)/n);
    sampleindex = [];
    for i = 1:length(sampleslec)
        index_t = ((sampleslec(i)-1)*samplepoints+1):(sampleslec(i)*samplepoints);
        sampleindex = cat(1,sampleindex,index_t);
    end

    [dataset_f,position] = manualline(wave_sle,wave,dataset);
    dataselc = dataset_f(sampleindex,:);
end
