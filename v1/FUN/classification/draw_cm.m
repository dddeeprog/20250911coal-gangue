function draw_cm(mat)
%%
% 参数：mat-矩阵；tick-要在坐标轴上显示的label向量，例如{‘label_1’,‘label_2’…}
%
    %%
    imagesc(mat); %# 绘彩色图
    colormap(flipud(hot));
    colorbar;
    num_class = size(mat,1);
    midValue = mean(get(gca, 'CLim'));
    [x,y] = meshgrid(0:num_class+1);
    title('Confusion Matrix','FontSize', 11, 'FontWeight', 'normal');
    xlabel('Predicted Labels','FontSize', 11, 'FontWeight', 'normal');
    ylabel('True Labels','FontSize', 11, 'FontWeight', 'normal');
    %% 
    for i=1:num_class
        for j=1:num_class
            if not (isnan(mat(i, j)))
            textStrings = num2str(mat(i, j),'%d');
%                 if mat(i, j) < 1
%                 textStrings = textStrings(2: 3);
%                 elseif mat(i, j) > 90
%                 textStrings = textStrings(2: 4);
%                 else
%                 textStrings = textStrings;
%                 end
                textStrings = strtrim(cellstr(textStrings));
                hStrings = text(j,i,textStrings,'HorizontalAlignment' , 'center', 'FontSize', 11, 'FontWeight', 'normal');
                textColors = repmat(mat(i, j) > midValue,1,3);
                %改变test的颜色，在黑cell里显示白色
                set(hStrings,{ 'Color' },num2cell(textColors,2)); %# Change the text colors
            end
        end
    end
end