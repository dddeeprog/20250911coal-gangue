%{
HWH-20220611
https://pubs.rsc.org/en/content/articlehtml/2015/an/c4an01061b
所提出的arPLS（非对称重加权惩罚最小二乘法）方法可以归纳为一种算法
有两个参数：p表示不对称，λ表示平滑度。两者都必须根据手头的数据进行调整。
我们发现，一般来说，0.001≤p≤0.1是一个不错的选择（对于具有正峰值的信号），
10^2≤λ≤10^9，但可能会出现例外。
sbuline =  baseline_arPLS(line,1000,0.01);\\
y 行

%}
function [subline,baseline] =  baseline_arPLS(y,lambda,ratio)
if nargin == 1
    lambda = 1000;
    ratio = 0.01;
elseif nargin == 2
    ratio = 0.01;
end
y = y';
%Estimate baseline with arPLS in Matlab 
N = length(y);
D = diff(speye(N),2);
H = lambda*(D'*D);
w = ones(N,1);
while true
    W = spdiags(w,0,N,N);
    %Cholesky decomposition 
    C = chol(W+H);
    z = C\(C'\(w.*y));
    d = y-z;
    % make d-，and get w^t with m and s 
    dn = d(d<0); 
    m = mean(dn); 
    s = std(dn);
    wt = 1./(1+exp(2*(d-(2*s-m))/s)); 
    %check exit condition and backup 
    if norm(w-wt)/norm(w)<ratio
        break; 
    end 
    w=wt;
end
z = z';
subline = y(:)-z(:);
subline = subline';
baseline = z(:);
baseline = baseline';
