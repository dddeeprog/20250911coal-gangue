%**************************************************************************
% 代码说明：峰值点寻找
% 输入：findpeak_covermax:根据波长覆盖范围内的最大值，默认波长在覆盖峰值内
%       选择波长范围7点 左右各三点之间的最大值
%       wave：对应的波长横坐标
%       data：数据矩阵，每一行为一个样本
%
% 输出：puku_peak:寻找的最大值
% peak_positions:寻找峰值的位置 按照数量最多的计算
% peakpos:寻找每一个 峰值的位置
% 拟合直线的斜率，截距
%**************************************************************************
function [puku_peak,peak_positions,peakpos] = findpeak_covermax(wave_slec,wave,data)
    peakpos = [];
    puku_peak = [];    
    for i = 1:size(data,1)
        line = data(i,:);
        peak_position = findpeak_covermax_singleline(wave_slec,wave,line);
        peakpos = cat(1,peakpos,peak_position);
        puku_peak = cat(1,puku_peak,line(peak_position));
        if mod(i,100)==0
            disp(i);
        end
    end
    [peak_positions,F,C] = mode(peakpos,1);%加1,避免单样本
end

function peak_position = findpeak_covermax_singleline(wave_slec,wave,line)
    peak_position = [];
    for i = 1:length(wave_slec)
        [~,right] = min(abs(wave_slec(i)-wave));
        % 寻找左右两个点的波长 位置 找到峰值点位置
        left = right - 1;
        
        % peak 峰值点位置
        if line(left) <= line(right)
            while line(right)<line(right+1)
                right = right+1; 
            end
            peak = right;
        elseif line(left) > line(right)
            while line(left-1)>line(left)
                left = left-1; 
            end
            peak = left;
        end
        peak_position = cat(2,peak_position,peak);
    end
end












