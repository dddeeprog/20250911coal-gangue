%{
20221122-DN
门宽、延时、能量等不同数据集的参数优化
可实现多个谱线的优化
dataset:原始数据集
wave：波长
wave_slec：选择的谱线
%}
function [mat_av,mat_std,noise_mat] = ParaOptimization(wave,dataset,name,wave_slec,label,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'wave',validScalarPosNum);
    addRequired(p,'dataset',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'label',validScalarPosNum);
    addOptional(p,'subbase',defaultcoe,validScalarPosNum);
    addOptional(p,'save',defaultcoe);
    parse(p,wave,dataset,wave_slec,label,varargin{:});
   
    % 判断wave中是否自带噪声范围，只有一列则不带噪声，需计算
    noisecal_flag = 0;
    if size(wave_slec,2) == 1
        noisecal_flag = 1;
    end
    % 是否需要去除基线
    subbase_flag = 0;
    if p.Results.subbase ~= 0
        subbase_flag = 1;
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
   %% 数据集 读取
    class_num = length(label);
    sample_num = round(size(dataset,1)/class_num);%共有n个种类
    mat_av = [];    %平均值
    mat_std = [];   %std
    for i = 1:class_num
        dataset_t = dataset((1:sample_num)+sample_num*(i-1),:);
        % 计算均值 方差
        [line,~] = manualline(wave_slec(:,1),wave,dataset_t);
        % 这里flag代表的是用哪一个标准差函数，如果取0，则代表除以N-1，如果是1代表的是除以N，
        % 第三个参数代表的是按照列求标准差还是按照行求标准差
        st = std(line,1,1);
        if subbase_flag == 0 %不需要去基线 
            line = mean(dataset_t,1);
            [av,~] = manualline(wave_slec(:,1),wave,line);
        elseif subbase_flag == 1 % 去基线 
            line = mean(dataset_t,1);
            [subline,~] =  baseline_arPLS(line);
            [av,~] = manualline(wave_slec(:,1),wave,subline);
        end
        mat_av = cat(1,mat_av,av);
        mat_std = cat(1,mat_std,st);
    end
    %% 噪声范围计算*******************************************************
    if noisecal_flag == 0 %自带噪声范围
        noise_mat = zeros(class_num,size(wave_slec,1));
        for i = 1:size(wave_slec,1)
            noise_class = [];
            for j = 1:class_num
                line = mean(dataset((1:sample_num)+sample_num*(j-1),:),1);
                [~,index_1] = min(abs(wave_slec(i,2)-wave));
                [~,index_2] = min(abs(wave_slec(i,3)-wave));
                noise_fragment = line(index_1:index_2);
                noise = std(noise_fragment);
                noise_class = cat(2,noise_class,noise);   %按列合并
            end
            noise_mat(:,i) = noise_class';
        end
    elseif noisecal_flag == 1% 只有波长一列 根据区域计算噪声范围
        noise_mat = zeros(class_num,size(wave_slec,1));
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
            noise_mat(:,i) = noise_class';
        end
    end
    %% 开始参数优化
    % Int_aver, Std_aver, Int_aver/Noise_aver,Std_aver/Int_aver*100;
    %result()
    
    % 强度优化
    for i = 1:size(wave_slec,1)
        if save
            clf;
            figure(1);
        else
            figure();
        end
            h = errorbar(label(:),mat_av(:,i),mat_std(:,i),'.','MarkerSize', 20);
            hold on;
            set(gca,'XLim',[label(1)-2,label(end)+2]);     %X轴的数据显示范围
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),'Int Optimize'],'x','Frequency (Hz)','y','Intensity (a.u.)')
            set(h, 'LineStyle', '-', 'Color', 'b','LineWidth', 1.2);
        % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{i},' ',num2str(wave_slec(i)),'Int'],'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',[name{i},' ',num2str(countlist+1),'Int'],'.png'], 'png')
            clf;
        end
        if save == 1
            clf;
        end
    end
    
    % RSD
%     for i = 1:size(wave_slec,1)
%         if save
%             clf;
%             figure(1);
%         else
%             figure();
%         end
%             
%         % 保存图像
%         if isnumeric(save) && save
%             saveas(gcf, [folder,[name{i},' ',num2str(wave_slec(i)),'RSD'],'.png'], 'png')
%             clf;
%         elseif ~isnumeric(save)
%             dirList=dir(folder); % 读取文件夹列表
%             countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
%             saveas(gcf, [folder,'/',[name{i},' ',num2str(countlist+1),'RSD'],'.png'], 'png')
%             clf;
%         end
%         if save == 1
%             clf;
%         end
%     end
    % SNR
    for i = 1:size(wave_slec,1)
        if save
            clf;
            figure(1);
        else
            figure();
        end
            yyaxis left
            h1 = plot( label(:), mat_av(:, i)./noise_mat(:, i),'.','MarkerSize', 20);
            hold on;
            set(gca,'YColor','k');
            set(gca,'XLim',[label(1)-2,label(end)+2]); %X轴的数据显示范围
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),' SNR'],'x','Frequency (Hz)','y','SNR')
            set(h1, 'LineStyle', '-', 'Color', 'k','LineWidth', 1.2);
            yyaxis right
            h2 = plot( label(:), 100*mat_std(:, i)./mat_av(:, i),'.','MarkerSize', 20);
            set(gca,'YColor','r');
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),' RSD'],'x','Frequency (Hz)','y','RSD (%)')
            set(h2, 'LineStyle', '-', 'Color', 'r','LineWidth', 1.2);
            % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{i},' ',num2str(wave_slec(i)),'SNR'],'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',[name{i},' ',num2str(countlist+1),'SNR'],'.png'], 'png')
            clf;
        end
        if save == 1
            clf;
        end
    end
end
    
    