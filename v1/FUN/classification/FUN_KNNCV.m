%**************************************************************************
% 代码说明：KNN 交叉验证 
% 输入：dataset : 一行为一个样本 最后一列为标签数据

% 输出： test_red,train_red
%        model_result = [R2,rmse,mape]
% distance:距离度量
%               'euclidean'       欧几里得距离，默认的
%               'cityblock'        绝对差的和
%               'cosine'           余弦   （作为向量处理）
%               'correlation'     相关距离  样本相关性（作为值序列处理）
%               'Hamming'      海明距离   不同的比特百分比（仅适用于二进制数据）
% rule:如何对样本进行分类
%               'nearest'  最近的K个的最多数
%               'random'    随机的最多数
%               'consensus' 共识规则
%**************************************************************************
function [re,zhi,zhimat] = FUN_KNNCV(data,label,K_fold,k)
    dataset = [data,label(:)];
    indices = crossvalind('Kfold', size(dataset, 1), K_fold);
    [~, data_c] = size(dataset);
    zhenzhi = [];
    yucezhi = [];
    %指定模型参数
%     k = 10;
    
    for i = 1 : K_fold

        % 获取第i份测试数据的索引逻辑值
        test_index = (indices == i);
        % 取反，获取第i份训练数据的索引逻辑值
        train_index = ~test_index;
        %获取数据集
        testData = dataset(test_index, 1 : data_c - 1);
        testLabel = dataset(test_index, data_c);
        trainData = dataset(train_index, 1 : data_c - 1);
        trainLabel = dataset(train_index, data_c);
%         mdl = fitcknn(trainData, trainLabel,'NumNeighbors',1);%k为对应的1,2,3,4.....
        mdl = fitcknn(trainData, trainLabel,...
                        'Distance', 'Euclidean',...
                        'NumNeighbors',1, ...
                        'DistanceWeight', 'Equal',...
                        'Standardize', true);%k为对应的1,2,3,4.....
        yhat_test = predict(mdl,testData);
        
        %模型预测的结果
        zhenzhi = [zhenzhi;testLabel(:)];
        yucezhi = [yucezhi;yhat_test(:)];
    end

    re = [zhenzhi,yucezhi];
    [zhi,zhimat,~] = cal2(zhenzhi,yucezhi,'n');

end
















