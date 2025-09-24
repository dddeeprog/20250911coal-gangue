%**************************************************************************
% 代码说明：实现三种线性拟合，无加权，直接加权，仪器加权
% 输入：data = [x,y,std]
%         flag = 'No_weighting'
%         flag = 'Direct_weighting'
%         flag = 'Instrumental'
% 输出：slop,jieju,R2,P_r,Adj_R2
% 拟合直线的斜率，截距
%**************************************************************************
function [slop,jieju,R2,P_r,Adj_R2,zhi,f] = Origin_linearFitpp(data,flag,p) 
    x = data(:,1);
    y = data(:,2);
    n = length(x);
    No_weight = repelem([1],n,1);
    if size(data,2)>2
        st = data(:,3);
    else
        st = No_weight;
    end
    direct_weight = st;
    yiqi_weight = (st.^-2)./mean(st.^-2);
    
    if strcmp(flag,'No_weighting')
        weight = No_weight;
    elseif strcmp(flag,'Direct_weighting')
        weight = direct_weight;
    elseif strcmp(flag,'Instrumental')
        weight = yiqi_weight;
    else
        error('输入不符合要求,请使用No_weighting,Direct_weighting或Instrumental')        
    end
    data = [x,y,weight];
    [slop,jieju] = weight_regress(data);
    f = @(x)slop*x+jieju;
    y_yu = f(x);
    data = [y,y_yu,weight];
    [R2,P_r,Adj_R2,zhi] = weight_R(data);
    
    if p == 1
        replot(zhi,slop,jieju,x,y,st,f);
    end
    disp('运行完成');
end

%求解回归方程
function [slop,jieju] = weight_regress(data)
    x = data(:,1);
    y = data(:,2);
    weight = data(:,3);
    x_y_av = sum(x.*y.*weight)/sum(weight);
    x_av = sum(x.*weight)/sum(weight);
    y_av = sum(y.*weight)/sum(weight);
    x2_av = sum(weight.*(x.^2))/sum(weight);
    slop = (x_y_av-x_av*y_av)/(x2_av-x_av^2);
    jieju = y_av - slop*x_av;
end

function [R2,P_r,Adj_R2,zhi] = weight_R(data)
    x = data(:,1);
    y = data(:,2);
    weight = data(:,3);
    rsd_av = mean(data(:,3)./data(:,2));
    x_av = sum(x.*weight)/sum(weight);
    ss_res = sum(weight.*((x-y).^2));
    ss_tot = sum(weight.*((x-x_av).^2));
    R2 = 1-ss_res/ss_tot;
    %pearson_r
    x_av = sum(x.*weight)/sum(weight);
    y_av = sum(y.*weight)/sum(weight);
    fenzi = sum(weight.*(x-x_av).*(y-y_av));
    fenmu1 = sqrt(sum(weight.*(x-x_av).^2));
    fenmu2 = sqrt(sum(weight.*(y-y_av).^2));
    P_r = fenzi/(fenmu1*fenmu2);
    %调整R2 一般多变量使用
    n = length(x);
%     Adj_R2 = 1-((1-R2^2)*(n-1)/(n-2));%不准确
    Adj_R2 = 1-(ss_res/(n-2))/(ss_tot/(n-1));

    % rmse mape mae 20220711 未加权重
    rmse = sqrt(mean((x-y).^2));
    rmse_j = sqrt(sum((x-y).^2)/(n-2));
    mape = mean(abs((y - x)./x))*100;
    mae = mean(abs(x-y));
    sse = sum((x-y).^2);
    % 输出
    zhi.R2 = R2;
    zhi.P_r = P_r;
    zhi.Adj_R2 = Adj_R2;
    zhi.rmse = rmse;
    zhi.mape = mape;
    zhi.mae = mae;
    zhi.sse = sse;
    zhi.rmse_j = rmse_j;
    zhi.rsd_av = rsd_av;

end

function replot(zhi,slop,jieju,x,y,st,f)
    txtc.R2 = zhi.R2;
    txtc.rmse = zhi.rmse;
    txtc.mae = zhi.mae;
    txtc.mape = zhi.mape;
    txtc.sse = zhi.sse;
    txtc.rsd = zhi.rsd_av;
    names = fieldnames(txtc);
    txt = {};
    for j = 1:length(names)
        txt{j} = [names{j},'=',num2str(txtc.(names{j}))];
    end
%     txt{end+1} = ['LOD=',num2str(LOD),'ppm'];
    gs = ['y=(',num2str(slop),')x+(',num2str(jieju),')'];


    % 绘图,展示相关的拟合结果
    x1 = linspace(min(x),max(x),10);
    y1 = f(x1);
    figure();
    h1 = errorbar(x,y,st,'.','MarkerSize', 20);   %注意'-o'中的h-e去掉后画出来的图是各个孤立的点
    hold on;
    h2 = plot(x1,y1);
    plotstyle('ptitle',flag,'x','Content (ppm)','y','Intensity (a.u.)');
    set(h1, 'LineStyle', 'None', 'Color', 'b','LineWidth', 1.2);
    set(h2,'Color', 'r','LineWidth', 1.2);
    text('string',txt,'Units','normalized','position',[0.65,0.25],"FontName",'times new roman','FontWeight','Bold')
    text('string',gs,'Units','normalized','position',[0.05,0.95],"FontName",'times new roman','FontWeight','Bold')
end