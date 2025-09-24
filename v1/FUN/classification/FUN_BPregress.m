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
function [perf,yp,zhi] = FUN_BPregress(data,label,hidelayer_param)
    % 转置
    x = data'; y = label';
    % 训练函数为 trainlm
    trainFcn = 'trainlm';

    % 初始化网络
    net = fitnet(hidelayer_param,trainFcn);

    % 设置比例
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;

    % 训练网络
    [net,tr] = train(net,x,y);

    % 计算所有训练样本预测值
    yp = sim(net,x);

    % 计算总体均方误差
    perf = perform(net,y,yp);
    zhi = cal([y(:),yp(:)],'R2');

    % 查看网络结构
%     view(net)
end