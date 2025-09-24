%**************************************************************************
% 代码说明：双高斯拟合获取拟合参数
% 输入：set : 一列为一个特征，行数为统计数
%       manuname：特征的名称，和列数一致
%       cance：用于测试优化的参数集

% 输出：test_red,train_red
%     red.R2 = R2;
%     red.result = result;
%     red.contr = contr;
%     red.percent_explained = percent_explained;
%**************************************************************************
function canshuall = FUN_gausfit(set,manuname,cance)

    cols = manuname;
    canshuall = [];
    for i = 1:size(set,2)
        line = set(:,i);
        histpic = histogram(line); 
        x = 1:histpic.NumBins;
        y = histpic.Values;
        clf
    %     eval(['x',num2str(i),'=','x(:);']);
    %     eval(['y',num2str(i),'=','y(:);']);
    %     figure(1)
    %     bar(x,y)
    %     plotstyle([cols{i},'元素分布与高斯拟合'],'范围编号','4500点分布频数');
    %     saveas(1,['C:\Users\hweihua\Desktop\hist',num2str(i),'.png'],'png'); %方法一 

        gaussian = @(a,x)a(1)*exp(-((x-a(2))/a(3)).^2)+a(4)*exp(-((x-a(5))/a(6)).^2);
    %##########################################################################
    %     a0 = gaus(i,:);
    %##########################################################################
        R2mat = [];
        for j = 1:size(cance,1)
            a0 = cance(j,:);
            [ahat,~,~,~,~] = nlinfit(x(:),y(:),gaussian,a0);%fit函数
            y_nihe = gaussian(ahat,x(:));
            zhi = cal([y(:),y_nihe(:)],'R2');
            R2mat = [R2mat,zhi.P_r];
        end
        [~,I] = max(R2mat);
        a0 = cance(I,:);
        
    %##########################################################################
        disp('正式拟合')
        [ahat,~,~,~,~] = nlinfit(x(:),y(:),gaussian,a0);%fit函数
        x2 = min(x):0.1:max(x);
        y_nihe2 = gaussian(ahat,x2);
        figure(1)
        bar(x,y)
        hold on 
        plot(x2,y_nihe2,'linewidth',3);
        plotstyle([cols{i},'元素分布与高斯拟合'],'范围编号','6000点分布频数');
        saveas(1,['E:\数据集\LIBS生成\20210603\铝合金fit\',num2str(i),'.png'],'png'); %方法一 
        canshuall = [canshuall;ahat];
        disp(i)
    end
end