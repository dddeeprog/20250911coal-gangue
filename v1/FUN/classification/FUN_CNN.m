

function zhi = FUN_CNN(dataset)
    inputLayer = imageInputLayer([50,19]);
    filtersize = [3,3];
    % numFilters = 16;
    middleLayers = [
        convolution2dLayer(filtersize, 32, 'Padding',1)
        reluLayer()
        maxPooling2dLayer(2,'Stride',2)
        dropoutLayer(0.2)
        convolution2dLayer(filtersize, 64, 'Padding','same')
        reluLayer()
        maxPooling2dLayer(2,'Stride',2)
        dropoutLayer(0.2)
        convolution2dLayer(filtersize, 128, 'Padding','same')
        reluLayer()
        maxPooling2dLayer(2,'Stride',2)
        dropoutLayer(0.2)
        ];
    %输出层
    finalLayers =[
        fullyConnectedLayer(128)
        reluLayer()
        fullyConnectedLayer(64)
        reluLayer()
        fullyConnectedLayer(32)
        reluLayer()
        fullyConnectedLayer(1)
        regressionLayer()
        ];
    layers = [
        inputLayer
        middleLayers
        finalLayers
        ];
    %**********************************************************************
    [trainData,trainLabel,testData,testLabel] = DataMaker(dataset,0.2);

    trainData3 = [];
    trainData2 = reshape(trainData,[size(trainData,1),50,19]);
    trainData2 = permute(trainData2,[2,3,1]);
    trainData3(:,:,1,:) = trainData2;

    testData3 = [];
    testData2 = reshape(testData,[size(testData,1),50,19]);
    testData2 = permute(testData2,[2,3,1]);
    testData3(:,:,1,:) = testData2;
    % show = trainData3(:,:,1,:);
    % show2 = squeeze(show);
    XTrain = trainData3;
    YTrain = trainLabel;
    %**********************************************************************
     %训练过程包括4步,每步可以使用单独的参数,也可以使用同一个参数
    miniBatchSize  = 16;
    validationFrequency = floor(numel(YTrain)/miniBatchSize);
    options = trainingOptions('sgdm', ...
        'MiniBatchSize',miniBatchSize, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.2, ...
        'LearnRateDropPeriod',20, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose',true);
%         'ExecutionEnvironment','gpu',...
    % options = [
    %     %第1步, Training a Region Proposa1 Network(RPN)
    %     trainingOptions('sgdm', 'MaxEpochs', 20,'MiniBatchSize',5,'InitiallearnRate', le-5,'Plots', 'training-progress' )
    %     %第2步, Training a Fast R-CNN Network using the RPN from step
    %     trainingOptions('sgdm', 'MaxEpochs', 20,'InitiallearnRate', le-5)
    %     %第3步,Re-training RPN using weight sharing with Fast R-CNN
    %     trainingOptions('sgdm', 'MaxEpochs', 20,'InitiallearnRate', le-6)
    %     %第4步,Re-training Fast R-CNN using updated RPN
    %     trainingOptions('sgdm', 'MaxEpochs', 20,'InitiallearnRate', le-6)
    %     ];
        %设置模型的本地存储
    doTrainingAndEval = 1;
    if doTrainingAndEval
        %训练R-CNN种经网络,神经网终工具箱提供了3个函数
        %(1) trainRCNNObjectDetector,训练快且检测慢,允许指定 proposalfcn
        %(2) trainFastRCNNObjectDetector,速度较快,允许指定 proposalFcn
        %(3) trainFasterRCNNObjectDetector,优化运行性能,不需要指定 proposalFcn
        net = trainNetwork(XTrain,YTrain,layers,options);
%         detector = trainFasterRCNNObjectDetector(trainData, layers, options,...
%         'NegativeoverlapRange',[0 0.3],...
%         'PositiveOverlapRange',[0.6 1],...
%         'BoxPyramidscale', 1.2);
    else
        %生加载已经训练好的神经网罗络
        net = data.net;    
    end
    %% 快速测试训练结果
    %运行检测器,输出目标位置和得分
    %(1)[bboxes, scores] = detect(detector, I);
    Xtest = testData3;
    YPredicted = predict(net,Xtest);

    result = [testLabel,YPredicted];
    zhi = cal(result,'R2');

    
end