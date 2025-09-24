%{
数据拟合 实现多项式拟合等

%}
classdef Fit < handle
    methods (Static = true)
        %使用polyfit实现线性拟合
        function [y_fit,coef] = xian(x,y)
            coef = polyfit(x,y,1);
    %         Fun = poly2sym(coef)    %显示拟合函数
            y_fit = polyval(coef,x);
        end
        function [y_fit,fitresult, gof,out] = xian1(test_label, pre_label)
            [xData, yData] = prepareCurveData( test_label, pre_label );
            % Set up fittype and options.
            ft = fittype( 'poly1' );
            % Fit model to data.
            [fitresult, gof,out] = fit( xData, yData, ft );
            y_fit = fitresult(test_label);% feval(fitresult,x)
            
        end
        function [y_fit,coef,resnorm] = xian2(x,y)
            a = [0 0];
%             f = @(a,x)a(1)+a(2)*(x).^0.5+a(3)*x+a(4)*(x).^2;
            f = @(a,x) a(1)*(x)+a(2);
            [coef,resnorm] = lsqcurvefit(f,a,x,y);
            y_fit = f(coef,x);
        end
        function [y_fit,coef] = xian3(x,y) 
            x1 = [x(:),ones(size(x(:),1),1)];
            coef = x1\y(:);
            f = @(a,x) a(1)*(x)+a(2);
            y_fit = f(coef,x);
        end


        % 'a*(x-b)^n'
        function [curve,gof] = nonxian(x, y)
            fo = fitoptions('Method','NonlinearLeastSquares',...
               'Lower',[0,0],...
               'Upper',[Inf,max(cdate)],...
               'StartPoint',[1 1]);
            ft = fittype('a*(x-b)^n','problem','n','options',fo);
            [curve,gof] = fit(x,y,ft,'problem',2);
        end
        

        
            
        
    end
end