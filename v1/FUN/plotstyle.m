%**************************************************************************
% 代码说明：plot绘图风格设置
% 输入：data:输入数据，每一行为一个特征，一列为一个样本

%
% 输出：perf:网络误差
% ylabel('\fontname{宋体}我喜欢 \fontname{Times New Roman}China');
%**************************************************************************
function plotstyle(varargin)
    p = inputParser;
    validScalarPosNum = @(x) isnumeric(x);
    addOptional(p,'ptitle','pic');
    addOptional(p,'x','x');
    addOptional(p,'y','y');
    addOptional(p,'px',200,validScalarPosNum);
    addOptional(p,'py',200,validScalarPosNum);
    addOptional(p,'w',400,validScalarPosNum);
    addOptional(p,'h',300,validScalarPosNum);

    parse(p,varargin{:});
    px = p.Results.px;
    py = p.Results.py;
    w = p.Results.w;
    h = p.Results.h;
    ptitle = p.Results.ptitle;
    x = p.Results.x;
    y = p.Results.y;
    title(ptitle);
    xlabel(x);
    ylabel(y);

    set(gcf,'position',[px,py,w,h])
%     set(gca,'FontSize',16,'FontName', 'times new roman');
%     set(gca,'TickLabelInterpreter','latex','FontSize',12,'FontWeight','Bold','FontName','times new roman');
    set(gca,'FontSize',12,'FontWeight','Bold','FontName','times new roman');
    set(gca,'Box','on')
    set(gca,'linewidth',1.5);
%     text( 'string',"(a) UB-OR", 'Units','normalized','position',[0.75,0.95],  'FontSize',14,'FontWeight','Bold','FontName','Times New Roman');  

    grid on
    set(gca,'gridlinestyle','--');

end