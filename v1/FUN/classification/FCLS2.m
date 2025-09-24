%{
求解方程 d = Cx  求x的非负解 s = N *x 
输入：
C:端元光谱 18000*p  一列为一个光谱
d:目标光谱 18000*1 一列为一个目标光谱
输出：
x(fengdu):系数矩阵 p*1
%}
function [fengdumat,RMSEmat] = FCLS2(C,d)
    fengdumat = zeros(size(C,2),size(d,2));
    RMSEmat = zeros(size(d,2));
    Delta = 1/1;
    for i = 1:size(d,2)
        s = d(:,i)*Delta;
        N = C*Delta;
        [fengdu, RMSE]= lsqnonneg(N,s);
        fengdumat(:,i) = fengdu;
        RMSEmat(i) = RMSE;
        if mod(i,100)==0
            disp(i)
        end
    end
end