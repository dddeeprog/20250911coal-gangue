%{
20220921-HWH 
lineshow 补充版本 
输入
datashow 绘制数据 m*n 500*20000 
samplename 样品名称 根据样品名称计算数据的样本数
elemname 对应选择谱线的名称
save 两种方式

varargin
sample_slec 默认绘制第一个样本 第一个样本的所有谱线 
feature_slec 默认绘制所有特征
same_y 1 将同一个特征的多个样本之间的纵坐标区间 设置成一致的
seek_mode 1 7点寻峰 2 固定位置 3 覆盖寻峰 默认2


应用实例：
lineshow2(datashow,wave,wave_slec,samplename,elemname,0)不保存
lineshow2(datashow,wave,wave_slec,samplename,elemname,1)保存
lineshow2(datashow,wave,wave_slec,samplename,elemname,'pic/t2/') 默认只画第一个样本
lineshow2(datashow,wave,wave_slec,samplename,elemname,1,'sample_slec',1:5,'same_y',1,'feature_slec',1:2)
lineshowfit(fitre,0,1:6,2:4) 选择2:4特征
%}

function lineshow2(datashow,wave,wave_slec,samplename,elemname,save,varargin)
    defaultcoe = 0;
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addRequired(p,'datashow');
    addRequired(p,'wave');
    addRequired(p,'wave_slec');
    addRequired(p,'samplename');
    addRequired(p,'elemname');
    addRequired(p,'save');
    addOptional(p,'sample_slec',defaultcoe);
    addOptional(p,'feature_slec',defaultcoe);
    addOptional(p,'same_y',defaultcoe);
    addOptional(p,'seek_mode',defaultcoe,validScalarPosNum);
    parse(p,datashow,wave,wave_slec,samplename,elemname,save,varargin{:});

    sample_slec = 1; %默认值 只选第一个样本
    if p.Results.sample_slec ~= 0 %不等于初始值 有输入 
        sample_slec = p.Results.sample_slec;
    end
    feature_slec = 1:length(wave_slec); % 默认值 所有特征
    if p.Results.feature_slec ~= 0
        feature_slec = p.Results.feature_slec;
    end
    same_y = 0; % 默认值 所有特征
    if p.Results.same_y ~= 0
        same_y = p.Results.same_y;
    end
    seek_mode = 2; % 默认值 固定值
    if p.Results.seek_mode ~= 0
        seek_mode = p.Results.seek_mode;
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
        if exist(folder,"dir")==0 %%判断文件夹是否存在
            mkdir(folder);  %%不存在时候，创建文件夹
        else
            disp('dir is exist'); %%如果文件夹存在，输出:dir is exist
        end
        disp('使用输入路径保存...')
    end
    % 基本信息计算
    samplenum = size(datashow,1)/length(samplename);
    
    % 寻峰
    [puku_peak,peak_positions,peakpos] = seekpeak(wave_slec,wave,datashow,seek_mode);
    width = 20;
    % 获取最大最小值区间 绘制相同特征的时候
    if same_y == 1
        sampleslec_index = [];
        for i = 1:length(sample_slec)
            index = ((sample_slec(i)-1)*samplenum+1):(sample_slec(i)*samplenum);
            sampleslec_index = cat(2,sampleslec_index,index);
        end
        dataslec = datashow(sampleslec_index,:);
        y_range = [];
        for i = 1:length(peak_positions)
            dataslec_p = dataslec(:,peak_positions(i)-width:peak_positions(i)+width);
            a = min(min(dataslec_p));
            b = max(max(dataslec_p));
            y_range = cat(2,y_range,[a;b]);
        end
    end

    % 按照顺序绘制

    for i = 1:length(sample_slec)
        sampleindex = ((sample_slec(i)-1)*samplenum+1):(sample_slec(i)*samplenum);
        datashow_t = datashow(sampleindex,:);

        for j = 1:length(feature_slec)
            j2 = feature_slec(j);
            % 计算曲线片段坐标  
            mul_line = datashow_t(:,peak_positions(j2)-width:peak_positions(j2)+width)';
            wave_line = wave(peak_positions(j2)-width:peak_positions(j2)+width);
            % 计算散点图坐标
            if seek_mode == 1 || seek_mode == 3
                x = wave(peakpos(sampleindex,j2));
                y = datashow_t(sub2ind(size(datashow_t),1:size(datashow_t,1),peakpos(sampleindex,j2)'));
            elseif seek_mode == 2 
                y = datashow_t(:,peak_positions(j2));
                x = repelem(wave(peak_positions(j2)),length(y),1);
            end

            % 绘制
            if save==0
                figure()
            else
                figure(1)
            end
            if size(mul_line,2) == 1%只有一条线
                plot(wave_line,mul_line,'b',LineWidth=1.5);
            else
                plot(wave_line,mul_line);
            end
            hold on;
            if size(mul_line,2) == 1%只有一条线
                scatter(x,y,'red','filled');
            elseif size(mul_line,2) ~= 1%
                scatter(x,y);
            end
            if same_y == 1 
%                 disp(y_range)
                ge = 0.1*(abs(y_range(2,j2)-y_range(1,j2)));
                ylim([y_range(1,j2)-ge,y_range(2,j2)+ge]);
            end
            set(gca,'XLim',[wave_line(1),wave_line(end)]);%X轴的数据显示范围
            plotstyle('ptitle',[samplename{sample_slec(i)},' - ',elemname{j2},num2str(wave_slec(j2))],'x','wavelength(nm)','y','Intensity(a.u.)')
            
            % 保存图像
            if isnumeric(save) && save
                saveas(gcf, [folder,[samplename{sample_slec(i)},'-',elemname{j2},num2str(wave_slec(j2))],'.png'], 'png')
                clf;
            elseif ~isnumeric(save)
                dirList=dir(folder); % 读取文件夹列表
                countlist = length(dirList)-2; %文件夹中文件数量，需减去2，有两个空文件
                saveas(gcf, [folder,'/',num2str(countlist+1),'.png'], 'png')
                clf;
            end
            disp([num2str(sample_slec(i)),'-',num2str(feature_slec(j))]);
        end
    end

end


function [puku_peak,peak_positions,peakpos] = seekpeak(wave_slec,wave,datashow,flag)
    if flag == 1
        [puku_peak,peak_positions,peakpos] = findpeak_7points(wave_slec,wave,datashow);
    elseif flag == 2
        peak_positions = [];
        puku_peak = [];
        for i = 1:length(wave_slec)
            [~,pos] = min(abs(wave_slec(i)-wave));
            peak_positions = cat(1,peak_positions,pos);
            peakpos = peak_positions;
            puku_peak = cat(2,puku_peak,datashow(:,pos));
        end
    elseif flag == 3
        [puku_peak,peak_positions,peakpos] = findpeak_covermax(wave_slec,wave,datashow);

    end
end












