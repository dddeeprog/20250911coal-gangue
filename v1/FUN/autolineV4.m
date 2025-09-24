%{ 
**************************************************************************
代码说明：自动选线 
1 cover 寻峰 寻找峰值点的位置
2 判断是否是峰值，宽约束条件 或条件 有一条满足即可
    2.1 连续两点下降 左侧或者右侧 重叠峰
    2.2 最大值点的位置比最小20%点的均值大3倍
3 根据计算结果，获取重复的点 （一个峰值下对应两条及以上谱线）判断到底是那一条
    3.1 规则是从上往下判断，存在重复则去重，然后重新判断，直到没有重复值
    3.2 获取重复的两条谱线 NIST库里最接近的 相邻的左侧或者右侧谱线的实际强度，比较两者强度值，大的优先级更高
        NIST库一般会有存在的谱线，如果没有被覆盖在波长范围 则为0
    3.3 相邻谱线的间距，如果在该线的附近（一定 指定范围）有很多条该元素的线存在（或者存在的数量更多），优先级更高。
    3.4 理论强度差异值过大 100倍以上 

输出值一定要标明 去重的线

代码一定要携带 autolineV4.mat 使用
输入： 
    line：待选线的谱线
输出：
    
20220915 - hwh
待改进：
    三线形式约束条件去重线
    输出去重线
**************************************************************************
%}
function [peaks_find,elemnames] = autolineV4(line,wave,eleslec_names)
%     line_sum = baseline_arPLS(line);
    line_sum = line;
    load('autolineV4.mat');
    strong = 0;
    if strong == 0
        ASD = ASD(ASD(:,5)==1,:);
    elseif strong == 1
        % use the strong line 
    end
    
    elexu_sle = elexu(ismember(elenames,eleslec_names),1);
    ASD_sle = ASD(ismember(ASD(:,1),elexu_sle),:);
    ASD_sle_inside = ASD_sle(ASD_sle(:,3)>wave(1)+0.1 & ASD_sle(:,3)<wave(end)-0.1,:); 
    
    %% 根据ASD谱线的波长进行选峰
    ASD_re = [];
    peaks_find = [];
    for i = 1:size(ASD_sle_inside,1)
        wave_slec = ASD_sle_inside(i,3);
        [puku_peak,peak_positions,peakpos] = findpeak_covermax(wave_slec,wave,line_sum);
        
        p = peak_positions;
        n = 2;
        % 左侧连续两点下降
        line_p = line_sum(p-n:p);
        condition1 = all(diff(line_p)>0);%
        % 右侧连续两点下降
        line_p = line_sum(p:p+n);
        condition2 = all(diff(line_p)<0);%
        % 三倍信号
        bei = 2;
        li = sort(line_sum(p-15:p+15));
        condition3 = line_sum(p)>(bei*mean(li(1:10)));
        if (condition1 || condition2) && condition3
%         if (condition1 || condition2)
            ASD_re = cat(1,ASD_re,ASD_sle_inside(i,:));
            peaks_find = cat(1,peaks_find,[ASD_sle_inside(i,1:4),puku_peak,wave(peak_positions),peak_positions]);
        end
    end
    disp('选峰完成')
    %% 去重
%     disp(peaks_find)
%     disp('dv')
    i = 1;
    while sum(isMultiple(peaks_find(:,end)))~=0
%         disp(i)
        T = isMultiple(peaks_find(:,end));
        mult_opos = find(T,1);
        index = find(peaks_find(:,end)==peaks_find(mult_opos,end),2); %表征重复行的位置 每次只计算2个重复的值
        
        mult_values = peaks_find(index,:);
        
        if mult_values(1,1) == mult_values(2,1) %原子序数相同 保留相对值较大 保留:更靠近中心
            if mult_values(1,4) >= mult_values(2,4)
                peaks_find(index(2),:) = [];
            else
                peaks_find(index(1),:) = [];
            end
        
        elseif mult_values(1,1) ~= mult_values(2,1) % 原子序数不同 不同的元素
            c01 = abs(mult_values(1,3)-mult_values(1,6))<0.15*abs(mult_values(2,3)-mult_values(2,6));
            c02 = abs(mult_values(2,3)-mult_values(2,6))<0.15*abs(mult_values(1,3)-mult_values(1,6));
            c1 = abs(mult_values(1,3)-mult_values(1,6))<0.8*abs(mult_values(2,3)-mult_values(2,6));
            c2 = abs(mult_values(2,3)-mult_values(2,6))<0.8*abs(mult_values(1,3)-mult_values(1,6));
            c3 = mult_values(1,4) >= 2*mult_values(2,4); %强度差异较大
            c4 = mult_values(1,4) <= 2*mult_values(2,4);
            if c01
                peaks_find(index(2),:) = [];
            elseif c02
                peaks_find(index(1),:) = [];
            elseif c1 && c3
                peaks_find(index(2),:) = [];
            elseif c2 && c4
                peaks_find(index(1),:) = [];
            elseif c3
                peaks_find(index(2),:) = [];
            elseif c4
                peaks_find(index(1),:) = [];
    
            else %其他 根据边上相邻两条线的强度比值进行计算
                try
                    if peaks_find(index(1)-1,1) == peaks_find(index(1),1) && peaks_find(index(1)+1,1) == peaks_find(index(1),1) %和前一个靠近的是相同元素
                        k1 = peaks_find(index(1),5)/mean([peaks_find(index(1)-1,5),peaks_find(index(1)+1,5)]);
                    elseif peaks_find(index(1)-1,1) == peaks_find(index(1),1)
                        k1 = peaks_find(index(1),5)/mean(peaks_find(index(1)-1,5));
                    elseif peaks_find(index(1)+1,1) == peaks_find(index(1),1)
                        k1 = peaks_find(index(1),5)/mean(peaks_find(index(1)+1,5));
                    end
                catch
                    k1 = 1;%孤立的线 只有一条
                end
                try
                    if peaks_find(index(2)-1,1) == peaks_find(index(2),1) && peaks_find(index(2)+1,1) == peaks_find(index(2),1) %和前一个靠近的是相同元素
                        k2 = peaks_find(index(2),5)/mean([peaks_find(index(2)-1,5),peaks_find(index(2)+1,5)]);
                    elseif peaks_find(index(2)-1,1) == peaks_find(index(2),1)
                        k2 = peaks_find(index(2),5)/mean(peaks_find(index(2)-1,5));
                    elseif peaks_find(index(2)+1,1) == peaks_find(index(2),1)
                        k2 = peaks_find(index(2),5)/mean(peaks_find(index(2)+1,5));
                    end
                catch 
                    k2 = 1;%孤立的线 只有一条
                end
                % 
                if k1 ==1 && k2 ~= 1 %把只有一条线的直接删了
                    peaks_find(index(1),:) = [];
                elseif k1 ~=1 && k2 == 1
                    peaks_find(index(2),:) = [];
                elseif k1>=k2
                    peaks_find(index(1),:) = [];
                elseif k1<=k2
                    peaks_find(index(2),:) = [];
                else 
                    peaks_find(index(2),:) = [];
                end
            end
        end
        i = i+1;
    end
    disp('去重完成')
    %% 获取元素名称
    [tf,index] = ismember(peaks_find(:,1),elexu);
    elemnames = elenames(index);
end



function T = isMultiple(A)
% T = isMultiple(A)
% INPUT:  A: Numerical or CHAR array of any dimensions.
% OUTPUT: T: TRUE if element occurs multiple times anywhere in the array.
%
% Tested: Matlab 2009a, 2015b(32/64), 2016b, 2018b, Win7/10
% Author: Jan, Heidelberg, (C) 2021
% License: CC BY-SA 3.0, see: creativecommons.org/licenses/by-sa/3.0/

T        = false(size(A));
[S, idx] = sort(A(:).');
m        = [false, diff(S) == 0];
if any(m)        % Any equal elements found:
   m(strfind(m, [false, true])) = true;
   T(idx) = m;   % Resort to original order
end
end

