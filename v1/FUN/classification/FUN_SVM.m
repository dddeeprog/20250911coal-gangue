%**************************************************************************
% 代码说明：调用fitcecoc 和 fitcsvm 实现SVM的多分类和二分类
% 输入：data:输入数据，每一行为一个特征，一列为一个样本
% flag : 1 多分类 0 二分类
%**************************************************************************
function [re,zhi,zhimat] = FUN_SVM(data,label,K_fold)
    dataset = [data,label(:)];
    num = tabulate(label);
    if size(num,1) <= 2
        flag = 1;
    else 
        flag = 2;
    end
    indices = crossvalind('Kfold', size(dataset, 1), K_fold);
    [~, data_c] = size(dataset);
    
    zhenzhi = [];
    yucezhi = [];
    %指定模型参数
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
        
        if flag==1
            mdl = fitcsvm(trainData, trainLabel);
%             mdl = fitcsvm(trainData, trainLabel,'OptimizeHyperparameters','all');
        elseif flag==2
%             mdl = fitcecoc(trainData, trainLabel);
%             t = templateSVM('Standardize',true,'KernelFunction','RBF');%改成RBF，测试结果很坏
            t = templateSVM(...
                                    'KernelFunction', 'polynomial', ...
                                    'PolynomialOrder', 3, ...
                                    'KernelScale', 'auto', ...
                                    'BoxConstraint', 1, ...
                                    'Standardize', true);
            mdl = fitcecoc(trainData, trainLabel,'Learners',t);

        end
        yhat_test = predict(mdl,testData);
        %模型预测的结果
        zhenzhi = [zhenzhi;testLabel(:)];
        yucezhi = [yucezhi;yhat_test(:)];
    end
    re = [zhenzhi,yucezhi];
    [zhi,zhimat] = cal2(zhenzhi,yucezhi,'n');

end