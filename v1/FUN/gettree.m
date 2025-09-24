%{
20220608-hwh
- 从csv2mat当中独立出来
- 用于软件中读取结构，建立树等
%}

function topli = gettree(basepath)
    path = basepath;
    topli = {};
    % 获取文件结构 每级的数量 
    for i = 1:5
        top = dir(path);
        topnames = {top.name};
        [topnames,~] = nasort(topnames);
        topnames = topnames(3:end);
    
        foldnames = cell(length(topnames),1);
        filenames = cell(length(topnames),1);
        for j = 1:length(topnames)
            path_t = [path,'/',topnames{j}];
    %         disp(path_t)
            if isfolder(path_t)
                foldnames{j} = topnames{j};
    
            elseif isfile(path_t)%判断如果是文件夹，进行读取
                [pathstr, name, ext] = fileparts(path_t);
                if strcmp(ext,'.csv') %如果是csv 则认为是最后一层
                    filenames{j} = topnames{j};
                else %不是csv文件，认为是文件夹的其他文件如png
                    continue
                end
            end
        end
        foldnames(cellfun(@isempty,foldnames))=[];
        filenames(cellfun(@isempty,filenames))=[];
        if ~isempty(foldnames)
            topli{i} = foldnames;
            path = [path,'/',foldnames{1}];
        elseif ~isempty(filenames)
            topli{i} = filenames;
            break
        end
    end
end