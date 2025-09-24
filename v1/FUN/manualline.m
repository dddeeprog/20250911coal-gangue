%{
HWH-20220612
光谱手动选线，精确指定光谱的波长，选择与之最接近一个峰值点。
相比于7点寻峰，7点只需粗略指定标准波长，但有时候寻不准。
%}

function [line,position] = manualline(wave_sle,wave,data)
    position = zeros(1,length(wave_sle));
    for i = 1:length(wave_sle)
        [~,p] = min(abs(wave-wave_sle(i)));
        position(i) = p;
    end
    line = data(:,position);
end
