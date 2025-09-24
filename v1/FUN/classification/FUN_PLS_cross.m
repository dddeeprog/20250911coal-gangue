%**************************************************************************
% 代码说明：PLS定量分析+交叉验证 
% 输入： dataset : 一行为一个样本 最后一列为标签数据
%       testRatio：测试集的占比
%       K_fold：交叉验证的折数
%       ncomp：降维成分数
%       flag：是否需要标准化，'no_zscore'，'zscore'
% 输出： test_red,train_red
%        model_result = [R2,rmse,mape]
%**************************************************************************
function [test_read,s,betaPlsPara,stats] = FUN_PLS_cross(dataset,K_fold,ncomp,flag)
    
    if strcmp(flag,'no_zscore')
        %将数据样本随机分割为K_fold部分
        indices = crossvalind('Kfold', size(dataset, 1), K_fold);
        [~, data_c] = size(dataset);
        test_result = [];
        yucezhi = [];
        zhenzhi = [];
        rmsemat = [];
        betaPLS = []; % 存储系数矩阵

        for i = 1 : K_fold
            % 获取第i份测试数据的索引逻辑值
            test = (indices == i);
            % 取反，获取第i份训练数据的索引逻辑值
            train = ~test;
            %获取数据集
            testData = dataset(test, 1 : data_c - 1);
            testLabel = dataset(test, data_c);
            trainData = dataset(train, 1 : data_c - 1);
            trainLabel = dataset(train, data_c);
        % 使用数据的代码
            %模型训练
            [Xloadings,Yloadings,Xscores,Yscores,betaPLS_i,PLSPctVar,MSE,stats] = plsregress(trainData,trainLabel,ncomp);
            %测试集预测
            data_in = testData;
            yhat_test = repmat(betaPLS_i(1,:),size(data_in,1),1)+data_in*betaPLS_i(2:size(data_in,2)+1,:);
            %模型预测的结果
            zhenzhi = [zhenzhi;testLabel];
            yucezhi = [yucezhi;yhat_test];
            betaPLS = [betaPLS,betaPLS_i];

            %rmsecv
            rmse_t = sqrt(mean((testLabel-yhat_test).^2));
            rmsemat = [rmsemat;rmse_t];
        end
        %计算模型的效果
        s = [zhenzhi,yucezhi];
        test_read = pingjia(zhenzhi,yucezhi);
        min_local=find(min(rmsemat));
        betaPlsPara = betaPLS(:,min_local);
        rmsecv = mean(rmsemat);
        test_read = [test_read,rmsecv];
        
    elseif strcmp(flag,'zscore')
       %将数据样本随机分割为K_fold部分
        indices = crossvalind('Kfold', size(dataset, 1), K_fold);
        [~, data_c] = size(dataset);
        test_result = [];
        yucezhi = [];
        zhenzhi = [];
        rmsemat = [];

        for i = 1 : K_fold
            % 获取第i份测试数据的索引逻辑值
            test = (indices == i);
            % 取反，获取第i份训练数据的索引逻辑值
            train = ~test;
            %获取数据集
            testData = dataset(test, 1 : data_c - 1);
            testLabel = dataset(test, data_c);
            trainData = dataset(train, 1 : data_c - 1);
            trainLabel = dataset(train, data_c);
            %数据集标准化
            [train_zs,data_av,data_sig] = zscore(trainData);
            [label_zs,label_av,label_sig] = zscore(trainLabel);
            %模型训练
            [Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar,MSE,stats] = plsregress(train_zs,label_zs,ncomp);
            %系数反变换
            ux = data_av;uy = label_av;
            sigx = data_sig;sigy = label_sig;
            beta = [uy-ux./sigx*betaPLS(2:end,:).*sigy+betaPLS(1)*sigy;(1./sigx)'*sigy.*betaPLS(2:end,:)];
            %采用变换系数 原始测试集预测
            data_in = testData;
            yhat_test = repmat(beta(1,:),size(data_in,1),1)+data_in*beta(2:size(data_in,2)+1,:);

            zhenzhi = [zhenzhi;testLabel];
            yucezhi = [yucezhi;yhat_test];
            
            %rmsecv
            rmse_t = sqrt(mean((testLabel-yhat_test).^2));
            rmsemat = [rmsemat;rmse_t];
        end
        %计算模型的效果
        s = [zhenzhi,yucezhi];
        test_read = pingjia(zhenzhi,yucezhi);
        rmsecv = mean(rmsemat);
        test_read = [test_read,rmsecv];
    else
        error('输入不符合要求,请使用no_zscore,zscore');
    end

end


function red = pingjia(x,y)
    rmse = sqrt(mean((y-x).^2));
    mape = mean(abs((x - y)./x))*100;
    R2 = 1 - sum((x-y).^2)./sum((x-mean(x)).^2);
    red = [R2,rmse,mape];
end


