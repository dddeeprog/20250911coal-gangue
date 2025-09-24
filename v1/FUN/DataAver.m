%**************************************************************************
% 代码说明：对数据集每四行进行平均
% 输入：data 根据flag变化
% 输出：
%       对输入数据集按行进行平均
% 
%**************************************************************************
function [mat_av,mat_std] = DataAver(mat,n)
    av_r = floor(size(mat,1)/n);
    mat_av = zeros(av_r,size(mat,2));
    mat_std = ones(av_r,size(mat,2));
    for i = 1:av_r
        index = (i-1)*n+1:i*n;
        mat_av(i,:) = mean(mat(index,:),1);
        mat_std(i,:) = std(mat(index,:),0,1);   %按列求标准差  除以n-1
    end
end