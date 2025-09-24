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
function [perf,classes_yuce,yuce_detail,acc,true_num] = FUN_BP(data,label,hidelayer_param)
    %为避免使用的标签不是从1开始的连续整数标签，先进行转化
    label2 = label;
    label3 = label;
%     label_detail1 = tabulate(label2);
    d1 = unique(label2);
    for i = 1:length(d1)
        label3(label2==d1(i)) = i;
    end
%     label_detail2 = tabulate(label3);
    label3 = label3';
    n = length(unique(label3));% 一共有多少类
    label_onehot = full(ind2vec(label3,n)); % ind2vec():将ind标签转换成vec稀疏编码，再由full()转换成OneHotEncoding
    %% patternnet模式识别网络
    label = label_onehot;
    classes = vec2ind(label);
    net = patternnet(hidelayer_param);
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 10/100;
    % net.tr.epoch = 2000;
    net = train(net,data,label);
    view(net)
    label_yuce = net(data);
    perf = perform(net,label,label_yuce);
    classes_yuce = vec2ind(label_yuce);
    yuce_detail = tabulate(classes);

    true_num = sum(classes==classes_yuce);
    acc = true_num/(length(classes));


end