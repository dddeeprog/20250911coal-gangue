%**************************************************************************
% 代码说明：掩膜法实现块状选取
% 输入：yanmo:0：无数据区域，1：有数据区域
%         k:选取的面积边长
%         xuan：选取的个数
%
% 输出：yanmo_new:选取的新掩膜，包含几个覆盖区域
%**************************************************************************
function yanmo_new = FUN_yanmo(yanmo,k,xuan)
    [m,n] = size(yanmo);
    row = floor(m/k);
    col = floor(n/k);
    index = [];
    for i = 1:row
        for j = 1:col
            S = sum(sum(yanmo((i-1)*k+1:i*k,(j-1)*k+1:j*k)));
            if S == k*k
                index = [index;[i,j]];
            end
        end
    end
    
    randnum = randperm(size(index,1),xuan);
    randindex = index(randnum,:);
    %构建新掩膜
    yanmostr = [];
    yanmo_n = zeros(m,n);
    for i = 1:xuan
        yanmo_n((randindex(i,1)-1)*k+1:randindex(i,1)*k,(randindex(i,2)-1)*k+1:randindex(i,2)*k) = 1;
        eval(['yanmostr.y',num2str(i),' = yanmo_n;']);
    end
    yanmo_new = yanmostr;
end