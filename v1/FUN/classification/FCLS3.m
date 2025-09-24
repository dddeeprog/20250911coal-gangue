%{
lsqlin
求解方程 d = Cx  求x的非负解 s = N *x lsqlin
和为1 SCLS 非负NCLS 双约束
输入：

C:端元光谱 18000*p  一列为一个光谱
d:目标光谱 18000*1 一列为一个目标光谱
输出：
x(fengdu):系数矩阵 p*1
%}
function [fengdumat,RMSEmat] = FCLS3(C,d)
    fengdumat = zeros(size(C,2),size(d,2));
    RMSEmat = zeros(size(d,2));
    Delta = 1/10;
    for i = 1:size(d,2)
        s = d(:,i)*Delta;
        s(size(s,1)+1,1) = 1;
        N = C*Delta;
        N(size(C,1)+1,:) = 1;
        [fengdu, RMSE]= lsqlin(N,s);
        fengdumat(:,i) = fengdu;
        RMSEmat(i) = RMSE;
        if mod(i,100)==0
            disp(i)
        end
    end
end