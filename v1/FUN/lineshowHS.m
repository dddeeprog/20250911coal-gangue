function lineshowHS(datashow,wave,waveslec_in,samplepoints_in,sample_slec_in,feature_slec_in,saveflag)

    % 创建文件夹
    folder = ['pic/lineshow-',datestr(now,30),'/']; %%定义变量
    if isnumeric(saveflag)
        if exist(folder,"dir")==0 && saveflag %%判断文件夹是否存在
            mkdir(folder);  %%不存在时候，创建文件夹
        else
            disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
        end
    else 
        folder = saveflag;
        disp('使用输入路径保存...')
    end
    
    for i = sample_slec_in
        for j = feature_slec_in
            [~,f_index] = min(abs(wave-waveslec_in(j)));
            data_t = datashow(((i-1)*samplepoints_in+1):(i*samplepoints_in),f_index);
            figure();
            histfit(data_t)
            plotstyle('ptitle',[num2str(i),'-',num2str(j),'-',num2str(waveslec_in(j))],'x','width','y','height')
            hold on;
            if isnumeric(saveflag) && saveflag
                saveas(gcf, [folder,[num2str(i),'-',num2str(j)],'.png'], 'png')
            elseif ~isnumeric(saveflag)
                dirList=dir(folder); % 读取文件夹列表
                countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
                saveas(gcf, [folder,'/',num2str(countlist+1),'.png'], 'png')
            end
            if saveflag == 1
                clf;
            end
            disp([i,j])
        end
    end
