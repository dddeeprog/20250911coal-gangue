% 配合findpeak2（CSDN下载）使用，p为其输出值，包含序号，位置，峰值，半高宽，面积。
function findpeak2plot(wave,subline,p)
    X = [];
    Y = [];
    for i = 1:size(p,1)
        X = [X,[p(i,2)-p(i,4)/2,p(i,2)+p(i,4)/2,p(i,2)+p(i,4)/2,p(i,2)-p(i,4)/2]'];
        Y = [Y,[0,0,p(i,3)*1.2,p(i,3)*1.2]'];
    end
    plot(wave,subline)
    hold on
    scatter(p(:,2),p(:,3))
    plotstyle('ptitle','Autoline-findpeak2','x','wavelength(nm)','y','Intensity(a.u.)');
    patch('XData',X,'YData',Y,'facecolor','r','facealpha',0.2)
end