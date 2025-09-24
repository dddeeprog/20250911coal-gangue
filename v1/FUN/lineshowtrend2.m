%{
lineshowtrend 展示谱线的波动性 趋势
展示所选谱线信息,用于观察数据的选择结果 是否符合预期
    datashow 要展示的数据 一般为一个样本的多次测量
    wave  
    wave_slec 选择波长 最好首选 较为精确的位置 代码选择最接近的点
    name  绘图的名称
    save 1 保存
    flag  1 根据波长 7点寻峰 2 选择最近的点
    sortmode 1 则选择排序后绘制

Cuall = [198.001;200.034;203.638;203.774;204.426;213.665;218.002;219.275;221.847;224.736;324.754;327.390;510.619;515.397;521.869];
elem = Cuall;
name = {};
for i = 1:length(elem)
    t = num2str(elem(i));
    name{i} = [t,'nm'];
end
应用实例
    lineshowtrend(datashow,wave,elem,name,1,2,'sortmode',1)
%}

function lineshowtrend2(datashow,wave,wave_slec,name,save,FileNum,flag,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'datashow',validScalarPosNum);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'name');
    addRequired(p,'save',validScalarPosNum);
    addRequired(p,'FileNum',validScalarPosNum);
    addRequired(p,'flag',validScalarPosNum);
    addOptional(p,'sortmode',defaultcoe,validScalarPosNum);
    parse(p,datashow,wave,wave_slec,name,save,FileNum,flag,varargin{:});
    % 读取是否需要排序
    sortmode = 0;
    if p.Results.sortmode ~= 0 
        sortmode = p.Results.sortmode;
    end
    % 创建文件夹
    folder = ['pic/lineshow-',datestr(now,30),'/']; %%定义变量
    if isnumeric(save)
        if exist(folder,"dir")==0 && save %%判断文件夹是否存在
            mkdir(folder);  %%不存在时候，创建文件夹
        else
            disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
        end
    else 
        folder = save;
        disp('使用输入路径保存...')
    end
    % 寻峰相关信息
    if flag == 1
        [puku_peak,peak_positions,peakpos] = findpeak_7points(wave_slec,wave,datashow);
    elseif flag == 2
        peak_positions = [];
        for i = 1:length(wave_slec)
            [~,pos] = min(abs(wave_slec(i)-wave));
            peak_positions = cat(1,peak_positions,pos);
        end
    end

    disp(peak_positions)
    RSD = zeros(FileNum,1);
    for j = 1:length(peak_positions)
        if save==1
            figure(1)
        elseif save == 0
            figure()
        end
        line = datashow(:,peak_positions(j))';
        samplenum = size(line,2)/FileNum;
        for k = 1 : FileNum
            line_temp = line(samplenum*(k-1)+1:samplenum*k);
            RSD(k,1) = std(line_temp)/mean(line_temp);
%             txt = ['RSD = ',num2str(RSD(k,1))];
        end
        for k = 1 : FileNum
            % 绘图            
            if sortmode ~= 0
                line = sort(line);
            else
            end
            plot(round(1:samplenum),line(samplenum*(k-1)+1:samplenum*k),'LineWidth',1.5)
            hold on 
        end
%         legend(num2str(RSD(1,1)),num2str(RSD(2,1)),num2str(RSD(3,1)),num2str(RSD(4,1)),num2str(RSD(5,1)));
%       text('string',txt,'Units','normalized','position',[0.65,0.1],"FontName",'times new roman','FontWeight','Bold')
        plotstyle('ptitle',[name{j},' ',num2str(wave_slec(j)),'nm'],'x','width','y','Intensity(a.u.)')
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{j},' ',num2str(wave_slec(j)),'nm'],'.png'], 'png')
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',num2str(countlist+1),'.png'], 'png')
        end
        if save == 1
            clf;
        end
        disp(j)
    end
end




