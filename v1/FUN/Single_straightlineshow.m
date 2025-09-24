%{
20221122-DN
门宽、延时、能量等不同数据集的参数优化
可实现多个谱线的优化
dataset:原始数据集
wave：波长
wave_slec：选择的谱线
%}
function Single_straightlineshow(mat_av,mat_std,SNR_av,SNR_std,wave_slec,label,name,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'mat_av',validScalarPosNum);
    addRequired(p,'mat_std',validScalarPosNum);
    addRequired(p,'SNR_av',validScalarPosNum);
    addRequired(p,'SNR_std',validScalarPosNum);
    addRequired(p,'wave_slec',validScalarPosNum);
    addRequired(p,'label',validScalarPosNum);
    addOptional(p,'dataslec',defaultcoe,validScalarPosNum);
    addOptional(p,'save',defaultcoe);
    parse(p,mat_av,mat_std,SNR_av,SNR_std,wave_slec,label,varargin{:});
   
    dataslec_flag = 0;
    if p.Results.dataslec ~= 0
        slecrange = p.Results.dataslec;
        dataslec_flag = 1;
    end
    
    [~,index] = sort(label);
    mat_av = mat_av(index,:);
    mat_std = mat_std(index,:);
    SNR_av = SNR_av(index,:);
    SNR_std = SNR_std(index,:);
    
    %% 数据集 读取
    class_num = length(label);
    sample_num = round(size(mat_av,1)/class_num);%共有n个种类
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
        mat_av = mat_av(index,:);
        mat_std = mat_std(index,:);
        SNR_av = SNR_av(index,:);
        SNR_std = SNR_std(index,:);
        label = label(slecrange);
    end
    %% 
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
    % Int
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
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),'Int Optimize'],'x','Defocus (mm)','y','Intensity (a.u.)')
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

    % SNR and RSD
    for i = 1:size(wave_slec,1)
        if save
            clf;
            figure(1);
        else
            figure();
        end
            yyaxis left
%             SNR = mat_av(:, i)./noise_mat(:, i);
%             SNR = Datanorm(SNR');
%             SNR = SNR';
            h1 = errorbar( label(:), SNR_av(:,i),SNR_std(:,i),'.','MarkerSize', 20);
            hold on;
            set(gca,'YColor','k');
            set(gca,'XLim',[label(1)-2,label(end)+2]); %X轴的数据显示范围
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),' SBR'],'x','Defocus (mm)','y','SBR')
            set(h1, 'LineStyle', '-', 'Color', 'k','LineWidth', 1.2);
            yyaxis right
            h2 = plot( label(:), 100*mat_std(:, i)./abs(mat_av(:, i)),'.','MarkerSize', 20);
            set(gca,'YColor','r');
            plotstyle('ptitle',[name{i},' ',num2str(wave_slec(i)),' RSD'],'x','Defocus (mm)','y','RSD (%)')
            set(h2, 'LineStyle', '-', 'Color', 'r','LineWidth', 1.2);
            % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{i},' ',num2str(wave_slec(i)),'SBR'],'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',[name{i},' ',num2str(countlist+1),'SBR'],'.png'], 'png')
            clf;
        end
        if save == 1
            clf;
        end
    end
end
    
    