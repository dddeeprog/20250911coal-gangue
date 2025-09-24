%{
20220607-hwh
- 用于制作mat格式数据集
- 二级或者三级文件夹
- 当存在csv,则认为是最后一级文件夹，其他格式文件，png等可存在
- 存直接存在mat格式，直接加载

% 判断是否存在某个指定格式的文件
tops = dir([basepath,'/*.mat']);
matnum = length({tops.name});
if matnum>0 %存在mat文件
%判断是否存在指定文件
% if isfile([basepath,'/','*.mat']) %判断文件是否存在
% 追加变量
save('abc.mat','m'，'-append')

- 返回值为S 数据集的结构体
- filednames(S)可获得字段
- 子数据集的文件数量必须一致，
%}

function [S,wave,msg] = csv2mat(basepath)
    %判断是否存在dataset.mat 和 wave.mat文件，如果存在则认为已建立
    if isfile([basepath,'/','dataset.mat']) && isfile([basepath,'/','wave.mat'])%判断文件是否存在
        %若已经存在dataset.mat,认为已经做过
        msg = '已存在dataset.mat/wave.mat文件,直接加载...';
        S = load([basepath,'/','dataset.mat']);
        wavedatas = load([basepath,'/','wave.mat']);
        a = fieldnames(wavedatas);
        wave = wavedatas.(a{1});
    %只存在dataset.mat 不存在wave.mat文件 加载data 制作wave
    elseif isfile([basepath,'/','dataset.mat']) && ~isfile([basepath,'/','wave.mat'])
        msg = '存在dataset.mat,不存在wave.mat文件,制作加载...';
        S = load([basepath,'/','dataset.mat']);
        topli = gettree(basepath);
        wave = makewave(basepath,topli);

    % 其他情况，均制作data和wave
    else
        msg = '双制作加载...';
        %获取文件结构
        topli = gettree(basepath);
        %根据文件数据制作数据集
        [S,wave] = makedataset(basepath,topli);
%         S = load([basepath,'/','dataset.mat']);
    end
end

% 根据路径，读取并保存mat文件
function [S,wave] = makedataset(basepath,topli)
    try
        % 情况1 一级文件夹 As-Zn基板\1ppm\1-1.csv 
        if length(topli)==2 %一级目录，二级文件
            % 读取波长文件
            wavepath = [basepath,'/',topli{1}{1},'/',topli{2}{1}];
            file = readmatrix(wavepath);
            wave = file(:,1);
            save([basepath,'/','wave.mat'],'wave')

            len1 = length(topli{1});
            len2 = length(topli{2});
            dataset = zeros(len1*len2,length(wave));
            h = waitbar(0,'please wait...'); %打开进度条，命名是为了方便关闭
            %设置双缓存，为了防止在不断循环画动画的时候会产生闪烁的现象
            set(h,'doublebuffer','on');
            for i = 1:len1
                for j = 1:len2
                    line = readmatrix([basepath,'/',topli{1}{i},'/',topli{2}{j}]);
                    line = line(:,2)';
                    dataset(len2*(i-1)+j,:) = line;
                end
                str=['正在计算类别',num2str(i),'中...'];
                waitbar(i/len1,h,str);
            end
            save([basepath,'/','dataset.mat'],'dataset')
            S.dataset = dataset;
            close(h);

        % 情况2 两级文件夹 As-Zn基板\1ppm\1\1-1.csv 将1视为次数，生成多个mat
        elseif length(topli)==3%一级目录，二级子目录(激发的点数），三级文件
            % 读取波长文件
            wavepath = [basepath,'/',topli{1}{1},'/',topli{2}{1},'/',topli{3}{1}];
            file = readmatrix(wavepath);
            wave = file(:,1);
            save([basepath,'/','wave.mat'],'wave')

            len1 = length(topli{1});
            len2 = length(topli{2});
            len3 = length(topli{3});

            h = waitbar(0,'please wait...'); %打开进度条，命名是为了方便关闭
            %设置双缓存，为了防止在不断循环画动画的时候会产生闪烁的现象
            set(h,'doublebuffer','on');
            for num = 1:len2
                dataset = zeros(len1*len3,length(wave));
                for i = 1:len1
                    for j = 1:len3
                        line = readmatrix([basepath,'/',topli{1}{i},'/',topli{2}{num},'/',topli{3}{j}]);
                        line = line(:,2)';
                        dataset(len3*(i-1)+j,:) = line;
                    end
                    str=['正在计算点数',num2str(num),'-类别',num2str(i),'中...'];
                    waitbar((len1*(num-1)+i)/(len1*len2),h,str);
                end
                % 循环修改保存语句 添加多个
                eval(['dataset_',num2str(num),'=dataset;']);
                S.(['dataset_',num2str(num)]) = dataset;
                if isfile([basepath,'/','dataset.mat'])
                    save([basepath,'/','dataset.mat'],['dataset_',num2str(num)],'-append')
                else 
                    save([basepath,'/','dataset.mat'],['dataset_',num2str(num)])
                end
            end
            close(h);
        %其他类型数据结构
        else
            disp('未识别结构');
        end
    catch
        disp('数据结构不一致，读取错误')
    end
end

% 制作波长文件
function wave = makewave(basepath,topli)
    try
        % 情况1 一级文件夹 As-Zn基板\1ppm\1-1.csv 
        if length(topli)==2 %一级目录，二级文件
            % 读取波长文件
            wavepath = [basepath,'/',topli{1}{1},'/',topli{2}{1}];
            file = readmatrix(wavepath);
            wave = file(:,1);
            save([basepath,'/','wave.mat'],'wave')
        % 情况2 两级文件夹 As-Zn基板\1ppm\1\1-1.csv 将1视为次数，生成多个mat
        elseif length(topli)==3%一级目录，二级子目录(激发的点数），三级文件
            wavepath = [basepath,'/',topli{1}{1},'/',topli{2}{1},'/',topli{3}{1}];
            file = readmatrix(wavepath);
            wave = file(:,1);
            save([basepath,'/','wave.mat'],'wave')
        else
            disp('波长未识别结构');
        end
    catch
        disp('波长数据结构不一致，读取错误')
    end
end












