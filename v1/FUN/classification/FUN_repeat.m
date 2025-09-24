%{ 
**************************************************************************
代码说明：将函数作用在矩阵的每一行上 

输入： 
    dataset：多行的矩阵
    fun : 作用函数
输出：
    dataset_processed : 处理后的样本
**************************************************************************
%}
function dataset_processed = FUN_repeat(dataset,fun)
    dataset_processed = [];
    for i = 1:size(dataset,1)
        line = dataset(i,:);
        line_t = fun(line);
        dataset_processed = cat(1,dataset_processed,line_t);
        if mod(i,100)==100
            disp(i)
        end
    end
end
