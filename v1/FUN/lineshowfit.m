%{
20220908-HWH 
搭配YU_fitS使用 用于绘制 面积拟合的情况
slec 默认为1 即有很多行的样本数据 只绘制第一行样本
slec 可选择 2 2:5 之类
应用实例：
lineshowfit(fitre,0)
lineshowfit(fitre,1)
lineshowfit(fitre,0,5)

应用实例：
    lineshowfit(fitre,0,1:6) 不保存
    lineshowfit(fitre,1,1:6) 保存
lineshowfit(fitre,0) 默认只画第一个样本
lineshowfit(fitre,0，1:6) 
lineshowfit(fitre,0，1:6,2:4) 选择2:4特征
%}

function lineshowfit(fitre,save,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'fitre');
    addRequired(p,'save',validScalarPosNum);
    addOptional(p,'sample_slec',defaultcoe,validScalarPosNum);
    addOptional(p,'feature_slec',defaultcoe,validScalarPosNum);
    parse(p,fitre,save,varargin{:});

    wave = fitre.wave;
    datain = fitre.data_out;
    s_mat = fitre.s_mat;
    f_mat = fitre.f_mat;
    fitxy = fitre.fitxy;
    puku_start_end = fitre.puku_start_end;

    sample_slec = 1; %默认值 只选第一个样本
    if p.Results.sample_slec ~= 0 %不等于初始值 有输入 
        sample_slec = p.Results.sample_slec;
    end

    feature_slec = 1:size(s_mat,2); % 默认值 所有特征
    if p.Results.feature_slec ~= 0
        feature_slec = p.Results.feature_slec;
    end

    % 
    n = size(datain,1);
    if save == 1
        % 创建文件夹
        folder = ['pic/',datestr(now,30),'/']; %%定义变量
        if exist(folder,"dir")==0 %%判断文件夹是否存在
            mkdir(folder);  %%不存在时候，创建文件夹
        else
            disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
        end
    end

    % 绘制多个样品
    for i = 1:length(sample_slec)
        for j = 1:length(feature_slec)
            left = puku_start_end{1,sample_slec(i)}(feature_slec(j),1);
            right = puku_start_end{1,sample_slec(i)}(feature_slec(j),2);
            l = 2*left-right;
            r = 2*right-left;
%             l = left;
%             r = right;
            x = wave(l:r);
            y = datain(sample_slec(i),l:r);
            fitx = fitxy{1,sample_slec(i)}{1,feature_slec(j)}(:,1);
            fity = fitxy{1,sample_slec(i)}{1,feature_slec(j)}(:,2);
            S = s_mat(sample_slec(i),feature_slec(j));
            F = f_mat(sample_slec(i),feature_slec(j));
            replot(x,y,fitx,fity,S,F,sample_slec(i),feature_slec(j));
            if save
                saveas(gcf, [folder,[num2str(sample_slec(i)),'-',num2str(feature_slec(j))],'.png'], 'png');
                close;
            end
        end    
    end
end


function replot(x,y,fitx,fity,S,F,sample_num,feature_num)
    txtc.fitS = S;
    txtc.FWHM = F;
    names = fieldnames(txtc);
    txt = {};
    for j = 1:length(names)
        txt{j} = [names{j},'=',num2str(txtc.(names{j}))];
    end

    % 绘图,展示相关的拟合结果
    figure();
%     disp(x)
%     disp(y)
    h1 = plot(x,y);

    hold on;
    h2 = area(fitx(:),fity(:));
    plotstyle('ptitle',[num2str(sample_num),'-',num2str(feature_num)],'x','Wavelength (nm)','y','Intensity (a.u.)');
    set(h1, 'Color', 'b','LineWidth', 1.2);
    set(h2,'facecolor','r','facealpha',0.2);
    set(gca,'XLim',[x(1),x(end)]);
    text('string',txt,'Units','normalized','position',[0.65,0.25],"FontName",'times new roman','FontWeight','Bold')
end














