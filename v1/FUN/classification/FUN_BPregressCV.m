%**************************************************************************
% 代码说明：调用patternnet实现模式识别，BP网络
% 输入：data:输入数据，每一行为一个特征，一列为一个样本
%         label:标签，
%         hidelayer_param:隐含层的层数和节点数设置
%
% 输出：perf:网络误差
%         classes_yuce：预测类别
%         yuce_detail：预测详情
%         acc：准确率
%         true_num：正确个数
%**************************************************************************
function [s,rmsecv] = FUN_BPregressCV(data,label,hidelayer_param,K_fold)
    dataset = [data,label];
    indices = crossvalind('Kfold', size(dataset, 1), K_fold);%K折会自动打乱顺序
    [~, data_c] = size(dataset);
    yucezhi = [];
    zhenzhi = [];
    rmsemat = [];
    
    % 训练函数为 trainlm
    trainFcn = 'trainlm';
    % 初始化网络
    net = fitnet(hidelayer_param,trainFcn);
    % 设置比例
    net.divideParam.trainRatio = 90/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 0/100;
%     view(net)

    for i = 1 : K_fold
        disp(i)
        % 获取第i份测试数据的索引逻辑值
        test_index = (indices == i);
        % 取反，获取第i份训练数据的索引逻辑值
        train_index = ~test_index;
        %获取数据集
        testData = dataset(test_index, 1 : data_c - 1);
        testLabel = dataset(test_index, data_c);
        trainData = dataset(train_index, 1 : data_c - 1);
        trainLabel = dataset(train_index, data_c);
        % 训练网络
        net2 = train(net,trainData',trainLabel');
        % 计算所有训练样本预测值
        yhat_test = sim(net2,testData');

        %模型预测的结果
        zhenzhi = [zhenzhi;testLabel(:)];
        yucezhi = [yucezhi;yhat_test(:)];

        %rmsecv
        rmse_t = sqrt(mean((testLabel(:)-yhat_test(:)).^2));
        rmsemat = [rmsemat;rmse_t];
    end
    %计算模型的效果
    s = [zhenzhi,yucezhi];
    rmsecv = mean(rmsemat);

    % 查看网络结构
%     view(net)
end