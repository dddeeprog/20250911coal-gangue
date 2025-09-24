%**************************************************************************
% 代码说明：PLS定量分析 
% 输入：dataset : 一行为一个样本 最后一列为标签数据
%       testRatio：测试集的占比
%       ncomp：降维成分数
%       flag：是否需要标准化，'no_zscore'，'zscore'
% 输出：test_red,train_red
%     red.R2 = R2;
%     red.result = result;
%     red.contr = contr;
%     red.percent_explained = percent_explained;
%**************************************************************************
function [test_red,train_red] = FUN_PLS(dataset,testRatio,ncomp,flag)
    if strcmp(flag,'no_zscore')
        [trainData,trainLabel,testData,testLabel] = DataMaker3(dataset,testRatio);
        %模型训练
        [Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar,MSE,stats] = plsregress(trainData,trainLabel,ncomp);
        %测试集预测
        data_in = testData;
        yhat_test = repmat(betaPLS(1,:),size(data_in,1),1)+data_in*betaPLS(2:size(data_in,2)+1,:);
        test_red = pingjia(testLabel,yhat_test,PLSPctVar);
        %训练集预测
        data_in = trainData;
        yhat_train = repmat(betaPLS(1,:),size(data_in,1),1)+data_in*betaPLS(2:size(data_in,2)+1,:);
        train_red = pingjia(trainLabel,yhat_train,PLSPctVar);
      
    elseif strcmp(flag,'zscore')
        [trainData,trainLabel,testData,testLabel] = DataMaker3(dataset,testRatio);
        %训练集标准化
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
        test_red = pingjia(testLabel,yhat_test,PLSPctVar);
        %采用变换系数 原始训练集预测
        data_in = trainData;
        yhat_train = repmat(beta(1,:),size(data_in,1),1)+data_in*beta(2:size(data_in,2)+1,:);
        train_red = pingjia(trainLabel,yhat_train,PLSPctVar); 
    else
        error('输入不符合要求,请使用no_zscore,zscore');
    end
end
%%
function red = pingjia(x,y,PLSPctVar)
    Y_pre = y;
    Y = x;
    error = abs(Y_pre - Y) ./ Y;
    % 决定系数R^2
    R2 = 1 - sum((Y-Y_pre).^2)./sum((Y-mean(Y)).^2);
    % 结果对比
    result = [Y Y_pre error];
    contr = cumsum(PLSPctVar,2);
    percent_explained = 100 * PLSPctVar(2,:) / sum(PLSPctVar(2,:));
    red.R2 = R2;
    red.result = result;
    red.contr = contr;
    red.percent_explained = percent_explained;

end
