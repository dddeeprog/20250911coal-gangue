%{
lineshow 
根据给定的waveslec 配合findpeak_7points进行展示
函数用于对光谱片段进行绘制与观察，选线-观察
flag 1:自动寻找峰值点的观察 2：手动选择峰值点的观察
sz 数据的形状  列*行
save 1 保存 0 不保存
lineshowpic(datashow,wave,wave_slec,sz,name,0,2)
%}

function lineshowpic(datashow,wave,wave_slec,sz,label,name,save,flag)
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
    for j = 1:length(peak_positions)
        if save==1
            figure(1)
        elseif save == 0
            figure()
        end
        line = datashow(:,peak_positions(j))';
        
        % 绘图
        pic = reshape(line,sz(1),sz(2));
        imagesc(pic');
        colorbar;

        plotstyle('ptitle',[num2str(label),'ppm',name{j},' ',num2str(wave_slec(j))],'x','width','y','height')
        if isnumeric(save) && save
            saveas(gcf, [folder,[num2str(label),'ppm',name{j},' ',num2str(wave_slec(j))],'.png'], 'png')
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




