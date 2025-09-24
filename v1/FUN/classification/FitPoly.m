function [fitresult, gof,out] = FitPoly(test_label, pre_label)
%CREATEFIT(TEST_LABEL,PRE_LABEL)
%  Create a fit.
%
%  Data for 'untitled fit 1' fit:
%      X Input : test_label
%      Y Output: pre_label
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.
%
%  另请参阅 FIT, CFIT, SFIT.

%  由 MATLAB 于 31-Dec-2021 09:48:46 自动生成


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( test_label, pre_label );
% Set up fittype and options.
ft = fittype( 'poly1' );
% Fit model to data.
[fitresult, gof,out] = fit( xData, yData, ft );

% % Plot fit with data.
% figure( 'Name', 'untitled fit 1' );
% h = plot( fitresult, xData, yData );
% legend( h, 'pre_label vs. test_label', 'untitled fit 1', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % Label axes
% xlabel( 'test_label', 'Interpreter', 'none' );
% ylabel( 'pre_label', 'Interpreter', 'none' );
% grid on


