function [wave_Output,Speccal_Output] = FUN_Anova(SpecCal,wave,total_Num)
% 计算每列的方差
variances = var(SpecCal);

% 获取方差最大的前k个列的坐标
k = total_Num; % 设置要获取的前k个最大方差的列数
[~, sorted_indices] = sort(variances, 'descend');
Selected_wave = sorted_indices(1:k);
wave_Output = wave(Selected_wave,:);
Speccal_Output = SpecCal(:,Selected_wave);
end