%{
适用于图像绘制
%}
function straightlineshow(label,name,wave_slec,AllRSD,All_rsd_std,AllSNR,All_SNR_std,AllInt,All_Int_std,save)
    % 创建文件夹
    folder = ['pic/straightlineshow-',datestr(now,30),'/']; %%定义变量
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
    
    for m = 1 : size(AllRSD,2)
        if save
            clf;
            figure(1);
        else
            figure();
        end
            h = errorbar(label(:),AllInt(:,m),All_Int_std(:,m),'.','MarkerSize', 20);
            hold on;
            set(gca,'XLim',[label(1)-2,label(end)+2]);     %X轴的数据显示范围
            plotstyle('ptitle',[name{m},' ',num2str(wave_slec(m)),'Int Optimize'],'x','Frequency (Hz)','y','Intensity (a.u.)')
            set(h, 'LineStyle', '-', 'Color', 'b','LineWidth', 1.2);
        % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{m},' ',num2str(wave_slec(m)),'Int'],'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',[name{m},' ',num2str(countlist+1),'Int'],'.png'], 'png')
            clf;
        end
        if save == 1
            clf;
        end
    end
    for m = 1 : size(AllSNR,2)
        if save
            clf;
            figure(1);
        else
            figure();
        end
            yyaxis left
            h1 = errorbar(label,AllSNR(:,m),All_SNR_std(:,m),'.','MarkerSize', 20);
            hold on;
            set(gca,'YColor','k');
            set(gca,'XLim',[label(1)-2,label(end)+2]); %X轴的数据显示范围
            plotstyle('ptitle',[name{m},' ',num2str(wave_slec(m)),' SNR'],'x','Defocus (mm)','y','SNR')
            set(h1, 'LineStyle', '-', 'Color', 'k','LineWidth', 1.2);
            yyaxis right
            h2 = errorbar( label(:), AllRSD(:,m),All_rsd_std(:,m),'.','MarkerSize', 20);
            set(gca,'YColor','r');
            plotstyle('ptitle',[name{m},' ',num2str(wave_slec(m)),' RSD'],'x','Defocus (mm)','y','RSD (%)')
            set(h2, 'LineStyle', '-', 'Color', 'r','LineWidth', 1.2);
            % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,[name{m},' ',num2str(wave_slec(m)),'SNR'],'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',[name{m},' ',num2str(countlist+1),'SNR'],'.png'], 'png')
            clf;
        end
        if save == 1
            clf;
        end        
    end
    
end