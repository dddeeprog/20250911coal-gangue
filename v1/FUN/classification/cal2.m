%     display('Accuracy   : The ratio of correctly classfied to all');
%     display('Precision  : True Positive/(True Positive + False Positve)');
%     display('Recall     : True Positive/(True Positive + False Negative');
%     display('F1 Score   : Harmonic Mean - (2*Precision*Recall)/(Precision+Recall)');
%     display('F0.5 Score : Weight more emphasize on Precision than Recall - (1+p^2)*(Precision*Recall)/(p^2*Precision*Recall)');
%     display('F2 Score   : Weight more emphasize on Recall than Precision - (1+p^2)*(Precision*Recall)/(p^2*Precision*Recall)');
% x为真实值，y为预测值
function [zhi2,zhi,zhimat] = cal2(actual,pred,flag)
    actual = actual(:);
    pred = pred(:);
    [confMat,~] = confusionmat(actual,pred);
    for i =1:size(confMat,1)
        recall(i)=confMat(i,i)/sum(confMat(i,:));
        precision(i)=confMat(i,i)/sum(confMat(:,i));
        F1(i) = 2*recall(i)*precision(i)/(recall(i)+precision(i));
        F2(i) = (1+2^2)*(precision(i)*recall(i))/(((2^2)*precision(i))+recall(i));
        F05(i) = (1+0.5^2)*(precision(i)*recall(i))/(((0.5^2)*precision(i))+recall(i));
    end
    zhimat = [recall',precision',F1',F2',F05'];
    acc = sum(actual==pred)/length(pred);
    recall_Ove=sum(recall)/size(confMat,1);
    precision_Ove=sum(precision)/size(confMat,1);
    F1_Ove=sum(F1)/size(confMat,1);
    F2_Ove=sum(F2)/size(confMat,1);
    F05_Ove=sum(F05)/size(confMat,1);
    zhi.acc = acc;
    zhi.recall = recall_Ove;
    zhi.precision = precision_Ove;
    zhi.F1 = F1_Ove;
    zhi.F2 = F2_Ove;
    zhi.F05 = F05_Ove;
    zhi2 = [zhi.acc,zhi.recall,zhi.precision,zhi.F1,zhi.F2,zhi.F05];
    if strcmp(flag,'y')
        confusionchart(actual,pred,'RowSummary','row-normalized','ColumnSummary','column-normalized');
    else
        
    end
end