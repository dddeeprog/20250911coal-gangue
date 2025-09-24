function [fitresult, gof] = Fit_chazhi(x, line)
%CREATEFIT(X,LINE)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : x
%      Y Output: line
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 23-Nov-2021 21:19:54 自动生成


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( x, line );

% Set up fittype and options.
ft = 'nearestinterp';

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, 'Normalize', 'on' );

% Plot fit with data.
figure( 'Name', 'untitled fit 1' );
h = plot( fitresult, xData, yData );
legend( h, 'line vs. x', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'line', 'Interpreter', 'none' );
grid on


