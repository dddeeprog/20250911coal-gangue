%{
20221128-HWH
定标函数以及相关参数指标计算Calibrate用于背包式
可实现对 单一元素 多个波长 谱线进行定标
dataset:原始数据集

wave_slec: 单一元素的多个波长，一行一个波长，用分号分隔 (手动选择峰值点位置更加精确) 
    Cd = [214.473,214,215; 226.560,227,228];手动给定每个元素的噪声范围
    Cd = [214.473;226.560;228.836];自动选择噪声范围

label: 元素的浓度和含量 与数据集的顺序必须相对应
    x = [1,2,4,6,8,10,15,0.1,0.2,0.4,0.6,0.8];

可选参数：

fitmode:0不加权 1：直接加权 2：仪器加权 默认不加权 方差全部为1

slec:选择特定的点进行定标，默认全部选择
    slec = [1:3,6],选择1,2,3,6这几个点进行定标计算。

subbase:去除基线后定量 默认不去基线，
    'yanslec',1
    为提升计算速度，先对单类取平均后，全谱去除基线 故只对av起作用 对std LOD(噪声） 无影响

应用实例：
Cd = [214.473,215,217;226.560,215,217;228.836,215,217];
label = [1,2,4,6,8,10,15,0.1,0.2,0.4,0.6,0.8];
re = Calibrate(wave,dataset1,Cd,label,'dataslec',1:6,'subbase',1,'fitmode',2,'save',1);

%}
function [re,mat_av,mat_std] = PortableLIBS_Calibrate(wave,dataset,wave_slec,label,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'dataset',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'label',validScalarPosNum);
    addOptional(p,'internal_wave',defaultcoe,validScalarPosNum);
    addOptional(p,'fitmode',defaultcoe,validScalarPosNum);
    addOptional(p,'dataslec',defaultcoe,validScalarPosNum);
    addOptional(p,'subbase',defaultcoe,validScalarPosNum);
    addOptional(p,'divideS',defaultcoe,validScalarPosNum);
    addOptional(p,'save',defaultcoe);
    parse(p,wave,dataset,wave_slec,label,varargin{:});
    % 判断wave中是否自带噪声范围，只有一列则不带噪声，需计算
    noisecal_flag = 0;
    if size(wave_slec,2) == 1
        noisecal_flag = 1;
    end
    % slec不为0，则说明需要按照slec进行选择定标
    dataslec_flag = 0;
    if p.Results.dataslec ~= 0
        slecrange = p.Results.dataslec;
        dataslec_flag = 1;
    end
    % 是否需要去除基线
    subbase_flag = 0;
    if p.Results.subbase ~= 0
        subbase_flag = 1;
    end
    % 线性拟合的方式
    fitmode = 0;
    if p.Results.fitmode ~= 0
        fitmode = p.Results.fitmode;
    end
    % 是否除面积大小
    divideS = 0;
    if p.Results.divideS ~= 0
        divideS = p.Results.divideS;
    end
    % 是否保存数据
    save = 0;
    if p.Results.save ~= 0
        % 创建文件夹
        save = p.Results.save;
        folder = ['pic/cal-',datestr(now,30),'/']; %%定义变量
        if isnumeric(save)
            if exist(folder,"dir")==0 && save %%判断文件夹是否存在
                mkdir(folder);  %%不存在时候，创建文件夹
            else
                disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
            end
        else 
            folder = save;
            if exist(folder,"dir")==0 %%判断文件夹是否存在
                mkdir(folder);  %%不存在时候，创建文件夹
            else
                disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
            end
            disp('使用输入路径保存...')
        end
        
    end
    % 是否内标
    internal_wave = 0;
    internal_flag = 0;
    if p.Results.internal_wave ~=0
        internal_wave = p.Results.internal_wave;
        internal_flag = 1;
    end 
    %% 数据集 读取
    class_num = length(label);
    sample_num = round(size(dataset,1)/class_num);%共有n个种类
    if dataslec_flag == 0
%         None;
    elseif dataslec_flag == 1
        % 根据label来选择需要的点 顺序点
        [~,index] = sort(label);
        range = sort(index(slecrange));%重新排序，尽量保持与原数据一致顺序
        slecrange = range;
        % 根据index选择数据和标签
        index =  [];
        for i = 1:length(slecrange)
            index_t = ((slecrange(i)-1)*sample_num+1):(slecrange(i)*sample_num);
            index = cat(2,index,index_t);
        end
        dataset = dataset(index,:);
        label = label(slecrange);
    end
    
    if internal_flag == 1
        class_num = length(label);
        sample_num = round(size(dataset,1)/class_num);%共有n个种类
        mat_av = [];
        mat_std = [];
        for i = 1:class_num
            dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
            % 计算均值 方差
            [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
            [internal_intensity,~] = manualline(internal_wave,wave,dataset_t);
            % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
            % 第三个参数代表的是按照列求标准差还是按照行求标准差
            line = line./internal_intensity;
            st = std(line,1,1);
            if subbase_flag == 0%不需要去基线
                line = mean(dataset_t,1);
                [av,~] = manualline(wave_slec(:,1),wave,line);
                [internal_intensity,~] = manualline(internal_wave,wave,line);
                av = av./internal_intensity;
            elseif subbase_flag == 1%需要去基线
                line = mean(dataset_t,1);
                [subline,~] =  baseline_arPLS(line);
                [av,~] = manualline(wave_slec(:,1),wave,subline);
                [internal_intensity,~] = manualline(internal_wave,wave,line);
                av = av./internal_intensity;
            end
            mat_av = cat(1,mat_av,av);
            mat_std = cat(1,mat_std,st);
        end
    else
        %% 数据计算基线去除等
        % 更新了dataset label 重新计算
        class_num = length(label);
        sample_num = round(size(dataset,1)/class_num);%共有n个种类
        mat_av = [];
        mat_std = [];
        for i = 1:class_num
            dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
            % 计算均值 方差
            [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
            % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
            % 第三个参数代表的是按照列求标准差还是按照行求标准差
            st = std(line,1,1);

            if subbase_flag == 0%不需要去基线
                line = mean(dataset_t,1);
                [av,~] = manualline(wave_slec(:,1),wave,line);
            elseif subbase_flag == 1%自带噪声范围
                line = mean(dataset_t,1);
                [subline,~] =  baseline_arPLS(line);
                [av,~] = manualline(wave_slec(:,1),wave,subline);
            end
            mat_av = cat(1,mat_av,av);
            mat_std = cat(1,mat_std,st);
        end
    end

    %% 噪声范围计算*******************************************************
    if noisecal_flag == 0%自带噪声范围
        noise_mat = [];
%        noise_mat = zeros(class_num,size(wave_slec,1));
        for i = 1:size(wave_slec,1)
            noise_class = [];
            for j = 1:class_num
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                [~,index_1] = min(abs(wave_slec(i,2)-wave));
                [~,index_2] = min(abs(wave_slec(i,3)-wave));
                noise_fragment = line(index_1:index_2);
                noise = std(noise_fragment);
                noise_class = cat(2,noise_class,noise);
            end
%            noise_class = noise_class';
%            noise_mat(:,i) = noise_class';
           noise = mean(noise_class);
           noise_mat = cat(1,noise_mat,noise);
        end
    elseif noisecal_flag == 1% 只有波长一列 根据区域计算噪声范围
        noise_mat = [];
        for i = 1:size(wave_slec,1)
            wave_t = wave_slec(i);
            [~,index] = min(abs(wave_t-wave));

            noise_class = [];
            for j = 1:class_num
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                noise_fragment = line(max(1,index-20):min(length(line),index+20));
                noise_fragment_sort = sort(noise_fragment);
                noise = std(noise_fragment_sort(1:round(0.8*length(noise_fragment_sort))));
                noise_class = cat(2,noise_class,noise);
            end
            noise = mean(noise_class);
            noise_mat = cat(1,noise_mat,noise);
        end
    end

    %% 开始定标
    re = [];
    for i = 1:size(wave_slec,1)
          data = [label(:),mat_av(:,i),mat_std(:,i)];
        if fitmode == 0
            flag = 'No_weighting';
        elseif fitmode == 1
            flag = 'Direct_weighting';
        elseif fitmode == 2
            flag = 'Instrumental';
        end
        [slop,jieju,R2,P_r,Adj_R2,zhi,f] = Origin_linearFitpp(data,flag,0);
        
        % 计算检出限 相关指标获取
        LOD = 3*noise_mat(i)/slop;
        RSD_av = mean(mat_std(:,i)./mat_av(:,i));
        txtc.R2 = zhi.R2;
        txtc.rmse = zhi.rmse;
        txtc.mae = zhi.mae;
        txtc.mape = zhi.mape;
        txtc.sse = zhi.sse;
        txtc.RSD_av = RSD_av;
        names = fieldnames(txtc);
        txt = {};
        for j = 1:length(names)
            txt{j} = [names{j},'=',num2str(txtc.(names{j}))];
        end
        txt{end+1} = ['LOD=',num2str(LOD),'ppm'];
        gs = ['y=(',num2str(slop),')x+(',num2str(jieju),')'];


        % 绘图,展示相关的拟合结果
        x1 = linspace(min(label),max(label),10);
        y1 = f(x1);
        if save
            clf;
            figure(1);
        else
            figure();
        end
        h1 = errorbar(label(:),mat_av(:,i),mat_std(:,i),'.','MarkerSize', 20);   %注意'-o'中的h-e去掉后画出来的图是各个孤立的点
        hold on;
        h2 = plot(x1,y1);
        plotstyle('ptitle',num2str(wave_slec(i)),'x','Content (ppm)','y','Intensity (a.u.)');
        set(h1, 'LineStyle', 'None', 'Color', 'b','LineWidth', 1.2);
        set(h2,'Color', 'r','LineWidth', 1.2);
        text('string',txt,'Units','normalized','position',[0.65,0.25],"FontName",'times new roman','FontWeight','Bold')
        text('string',gs,'Units','normalized','position',[0.05,0.95],"FontName",'times new roman','FontWeight','Bold')
        re = [re;[slop,jieju,R2]];

        % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,num2str(wave_slec(i)),'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',num2str(countlist+1),'.png'], 'png')
            clf;
        end

        if save == 1
            clf;
        end

        
    end

end
