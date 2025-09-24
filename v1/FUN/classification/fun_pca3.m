function [data_PCA, COEFF, sum_explained, n,latent1]=fun_pca3(data)
    %用percent_threshold决定保留xx%的贡献率
    percent_threshold=98;   %百分比阈值，用于决定保留的主成分个数；
%     data=zscore(data);  %归一化数据
    [COEFF,SCORE,latent,tsquared,explained,mu]=pca(data);
    latent1=100*latent/sum(latent);%将latent特征值总和统一为100，便于观察贡献率
    A=length(latent1);
    percents=0;                          %累积百分比
    for n=1:A
        percents=percents+latent1(n);
        if percents>percent_threshold
            break;
        end
    end
    data= bsxfun(@minus,data,mean(data,1));
    data_PCA=data*COEFF(:,1:n);
%     pareto(latent1);%调用matla画图 pareto仅绘制累积分布的前95%，因此y中的部分元素并未显示
%     xlabel('Principal Component');
%     ylabel('Variance Explained (%)');
%     % 图中的线表示的累积变量解释程度
%     print(gcf,'-dpng','PCA.png');
    sum_explained=sum(explained(1:n));