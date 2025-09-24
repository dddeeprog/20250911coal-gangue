%{
lineshow 对目标区域片段进行展示
默认选择其左右各20个点进行绘制
根据给定的waveslec 配合findpeak_7points进行展示
函数用于对光谱片段进行绘制与观察，选线-观察
flag 1:自动寻找峰值点的观察 2：手动选择峰值点的观察
%}

function lineshow(datashow,wave,wave_slec,name,save,flag)
    width = 20;
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
        if exist(folder,"dir")==0 %%判断文件夹是否存在
            mkdir(folder);  %%不存在时候，创建文件夹
        else
            disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
        end
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
    elseif flag == 3
        [puku_peak,peak_positions,peakpos] = findpeak_covermax(wave_slec,wave,datashow);
    end

    disp(peak_positions)
    for j = 1:length(peak_positions)
        if save==1
            figure(1)
        elseif save == 0
            figure()
        end
        mul_line = datashow(:,peak_positions(j)-width:peak_positions(j)+width)';
        wave_line = wave(peak_positions(j)-width:peak_positions(j)+width);
        %散点图绘制
        if flag == 1
            y = datashow(sub2ind(size(datashow),1:size(datashow,1),peakpos(:,j)'));
        elseif flag == 2
            y = datashow(:,peak_positions(j));
        end
        %绘图
        plot(wave_line,mul_line);
        hold on;
        if flag == 1
            scatter(wave(peakpos(:,j)),y);
        elseif flag == 2 
            scatter(repelem(wave(peak_positions(j)),length(y),1),y);
        end
        
        set(gca,'XLim',[wave_line(1),wave_line(end)]);%X轴的数据显示范围
        plotstyle('ptitle',[name{j},' ',num2str(wave_slec(j)),' nm'],'x','wavelength(nm)','y','Intensity(a.u.)')
        % 保存图像
        if isnumeric(save) && save
            saveas(gcf, [folder,'/',name{j},' ',num2str(wave_slec(j)),'.png'], 'png')
            clf;
        elseif ~isnumeric(save)
            dirList=dir(folder); % 读取文件夹列表
            countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
            saveas(gcf, [folder,'/',num2str(countlist+1),'.png'], 'png')
            clf;
        end
%         if all(save == 1 ,~isnumeric(save))
%             clf;
%         end
%         disp(j)
    end
end




