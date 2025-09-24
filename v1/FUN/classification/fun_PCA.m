%**************************************************************************
% 1）获取样本数据 X ，样本为行，特征为列。
    % 2）对样本数据中心化，得S（S = X的各列减去各列的均值）。
    % 3）求 S 的协方差矩阵 C = cov(S)
    % 4) 对协方差矩阵 C 进行特征分解 [P,Lambda] = eig(C);
    % 5）结束。
    % 1、输入参数 X 是一个 n 行 p 列的矩阵。每行代表一个样本观察数据，每列则代表一个属性，或特征。
    % 2、COEFF 就是所需要的特征向量组成的矩阵，是一个 p 行 p 列的矩阵，没列表示一个出成分向量，经常也称为（协方差矩阵的）特征向量。并且是按照对应特征值降序排列的。所以，如果只需要前 k 个主成分向量，可通过：COEFF(:,1:k) 来获得。
    % 3、SCORE 表示原数据在各主成分向量上的投影。但注意：是原数据经过中心化后在主成分向量上的投影。即通过：SCORE = x0*COEFF 求得。其中 x0 是中心平移后的 X（注意：是对维度进行中心平移，而非样本。），因此在重建时，就需要加上这个平均值了。
    % 4、latent 是一个列向量，表示特征值，并且按降序排列。
    % 5、tsquared Hotelling的每个观测值X的T平方统计量
    % 6、explained 由每个主成分解释的总方差的百分比
    % 7、mu 每个变量X的估计平均值<br>% x= bsxfun(@minus,x,mean(x,1));
%**************************************************************************
function [data_PCA,im,COEFF,SCORE,latent1,tsquared,explained,mu] = fun_PCA(data,k)
    x = zscore(data);  %归一化数据
%     x=data;
    [COEFF,SCORE,latent,tsquared,explained,mu]=pca(x);
    data_PCA=x*COEFF(:,1:k);
    latent1=100*latent/sum(latent);%将latent总和统一为100，便于观察贡献率
    im = sum(latent1(1:k));
%     pareto(latent1);%调用matla画图 pareto仅绘制累积分布的前95%，因此y中的部分元素并未显示
%     xlabel('Principal Component');
%     ylabel('Variance Explained (%)');
    % 图中的线表示的累积变量解释程度
    
    

    
    
    
    
    
    
    
    
    
    