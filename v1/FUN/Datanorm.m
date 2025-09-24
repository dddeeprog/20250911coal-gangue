%{
1. [Y,PS] = mapminmax(X,YMIN,YMAX)
2. [Y,PS] = mapminmax(X,FP)
3. Y = mapminmax('apply',X,PS)
4. X = mapminmax('reverse',Y,PS)

1.  [Y,PS] = mapstd(X,ymean,ystd)
2. [Y,PS] = mapstd(X,FP)
3. Y = mapstd('apply',X,PS)
4. X = mapstd('reverse',Y,PS)

%}

function Y = Datanorm(X)
    YMIN = 0;
    YMAX = 1;
    [Y,PS] = mapminmax(X,YMIN,YMAX);
end