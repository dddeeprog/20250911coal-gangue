%{ 
**************************************************************************
代码说明：根据寻峰结果选择波段

输入： 
    line：待选线的数据集 一行一个 绘制
    waveslec : 寻峰的结果 真实的峰值位置
输出：
    bopu：谱峰片段
    bopuwide:谱峰片段*1.5 左右扩展
    bopucell：cell格式的波谱
    bopuwidecell:谱峰片段*1.5 左右扩展
    
[~,eleslec_names] = xlsread('输入参数.xlsx',1);
[elem,peakindex,indexmat,indexmat_K] = autolineV1(line,lvhejin_wave,eleslec_names);
**************************************************************************
%}
function [bopu,bopuwave,re] = findbopu(line,wave,waveslec)
    bopu = [];
    bopuwave = [];
    bopuwide = [];
    bopuwidewave = [];

    bopucell = cell(1,length(waveslec));
    bopuwavecell = cell(1,length(waveslec));
    bopuwidecell = cell(1,length(waveslec));
    bopuwidewavecell = cell(1,length(waveslec));

    line = line(:);
    line = line';
    wave = wave(:);
    wave = wave';
    for i = 1:length(waveslec)
        [~,pos] = min(abs(waveslec(i)-wave));
        bopuindex = pos;
        while 1 %向左找到连续上升的两点
            pos = pos-1; 
            if pos==0
                break
            end
            if line(pos)>line(pos+1) && line(pos-1)>line(pos) %连续两点上升
                break
            end
            bopuindex = [pos,bopuindex];
        end
        [~,pos] = min(abs(waveslec(i)-wave));
        while 1 %向右找到连续上升的两点
            pos = pos+1;
            if pos==length(wave)
                break
            end
            if line(pos-1)<line(pos) && line(pos)<line(pos+1) %连续两点上升
                break
            end
            bopuindex = [bopuindex,pos];
        end
        % 映射原始数据的位置
        bopu = cat(2,bopu,line(bopuindex));
        bopuwave = cat(2,bopuwave,wave(bopuindex));
        bopucell{i} = line(bopuindex);
        bopuwavecell{i} = wave(bopuindex);
        
        % 计算加宽 或者变窄的
        K = 1.5;
        bupowidth_L = pos-bopuindex(1);
        bopuwidth_R = bopuindex(end)-pos;
        bupowidth_L_val = round(pos-bupowidth_L*K);
        if bupowidth_L_val<1
            bupowidth_L_val = 1;
        end
        bupowidth_R_val = round(pos+bopuwidth_R*K);
        if bupowidth_R_val>length(wave)
            bupowidth_R_val = length(wave);
        end
        bopuindex_val = bupowidth_L_val:bupowidth_R_val;

        bopuwide = cat(2,bopuwide,line(bopuindex_val));
        bopuwidewave = cat(2,bopuwidewave,wave(bopuindex_val));
        bopuwidecell{i} = line(bopuindex_val);
        bopuwidewavecell = wave(bopuindex_val);


    end
    re.bopucell = bopucell;
    re.bopuwavecell = bopuwavecell;
    re.bopuwide = bopuwide;
    re.bopuwidewave = bopuwidewave;
    re.bopuwidecell = bopuwidecell;
    re.bopuwidewavecell = bopuwidewavecell;
end






























