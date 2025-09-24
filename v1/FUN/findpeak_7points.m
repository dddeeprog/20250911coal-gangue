%**************************************************************************
% 代码说明：峰值点寻找
% 输入：findpeak_7points:7点寻峰
%       选择波长范围7点 左右各三点之间的最大值
%       wave：对应的波长横坐标
%       data：数据矩阵，每一行为一个样本
%
% 输出：puku_peak:寻找的最大值
% peak_positions:寻找峰值的位置 按照数量最多的计算
% peakpos:寻找每一个 峰值的位置
% 拟合直线的斜率，截距
%**************************************************************************
function [puku_peak,peak_positions,peakpos] = findpeak_7points(wave_sle,wave,data)
    peakpos = [];
    puku_peak = [];
    dianshu = 3;
    for i = 1:size(data,1)
        line = data(i,:);
        peak_positions = [];
        for j = 1:length(wave_sle)    
            [~,position] = min(abs(wave-wave_sle(j)));
            dianshu_xu = 1:dianshu;
            position_hou = position+dianshu_xu;
            position_qian = position-flip(dianshu_xu);

            position_set = [position_qian,position,position_hou];
            [~,peak_po] = max(line(position_set));
            peak_position = peak_po+position_set(1)-1; %找出在全谱中的位置
            peak_positions = [peak_positions,peak_position];
        end 
    %     peak_positions = unique(peak_positions);
        peak_line = line(peak_positions);
        puku_peak(i,:) = peak_line;
        peakpos = [peakpos;peak_positions];
        
        if mod(i,100)==0
%             disp(i);
        end
    end
    [peak_positions,F,C] = mode(peakpos,1);%加1,避免单样本
end