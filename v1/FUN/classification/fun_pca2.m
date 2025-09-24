function [data_PCA,sum_explained, COEFF ]=fun_pca2(data,k)
    % k:前k个主成分
    data=zscore(data);  %归一化数据
    [COEFF,SCORE,latent,tsquared,explained,mu]=pca(data);
    latent1=100*latent/sum(latent);%将latent特征值总和统一为100，便于观察贡献率
    data= bsxfun(@minus,data,mean(data,1));
    data_PCA=data*COEFF(:,1:k);
%     pareto(latent1);%调用matla画图 pareto仅绘制累积分布的前95%，因此y中的部分元素并未显示
%     xlabel('Principal Component');
%     ylabel('Variance Explained (%)');
%     % 图中的线表示的累积变量解释程度
%     print(gcf,'-dpng','PCA.png');
    sum_explained=sum(explained(1:k));


    