function pd = DataDist(flag)
%     flag : 'Normal'  https://ww2.mathworks.cn/help/stats/makedist.html
    mu = 0;
    sigma = 1;
    pd = makedist(flag,'mu',mu,'sigma',sigma);
end