%**************************************************************************
% 代码说明：调用fitcecoc 和 fitcsvm 实现SVM的多分类和二分类
% 输入：data:输入数据，每一行为一个特征，一列为一个样本

%**************************************************************************
function [re,zhi,zhimat] = FUN_RFclass(data,label,K_fold)
    dataset = [data,label(:)];
    indices = crossvalind('Kfold', size(dataset, 1), K_fold);
    [~, data_c] = size(dataset);
    zhenzhi = [];
    yucezhi = [];
    %指定模型参数5 2
    treenum = 50;
    leafnum = 10;
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
        
        mdl = TreeBagger(treenum,trainData,trainLabel,'Method','classification','OOBPredictorImportance','On',...
        'MinLeafSize',leafnum);
    
        yhat_test = predict(mdl,testData);
        %模型预测的结果
        zhenzhi = [zhenzhi;testLabel(:)];
        yucezhi = [yucezhi;yhat_test(:)];
%         disp(class(zhenzhi));
%         disp(class(yucezhi));
    end
%     yucezhi = str2num(char(yucezhi));
    yucezhi = str2num(cell2mat(yucezhi));
    zhenzhi = double(zhenzhi);
%     disp(class(yucezhi));
    re = [zhenzhi,yucezhi];
    [zhi,zhimat] = cal2(zhenzhi,yucezhi,'n');

end