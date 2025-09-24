%**************************************************************************
% 代码说明：RF 随机森林算法实现分类 回归 
% 输入：dataset : 一行为一个样本 最后一列为标签数据

% 输出： test_red,train_red
%        model_result = [R2,rmse,mape]
%**************************************************************************
function [RF_index,weight,e] = FUN_RF_FS(dataset)
    X = dataset(:,1:end-1);
    Y = dataset(:,end);
    treenum = 1000;
    leafnum = 5;
    t=cputime;
    mdl = TreeBagger(treenum,X,Y,'Method','classification','OOBPredictorImportance','On',...
    'MinLeafSize',leafnum);
%     mdl = TreeBagger(treenum,X,Y,'Method','classification','OOBPredictorImportance','On',...
%     'CategoricalPredictors',find(isCategorical == 1),...
%     'MinLeafSize',leafnum);
    weight = mdl.OOBPermutedPredictorDeltaError;
    e=cputime-t;
    [~,RF_index] = sort(weight,'descend');
end
