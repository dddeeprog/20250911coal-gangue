function [SubBG_Spectrum, Baseline] = SubBackground_ModelFree(Spectrum, W)
% 输入：
% Spectrum，光谱强度列向量
% 窗口大小，30~50
% 输出：
% SubBG_Spectrum，已经扣除基线的光谱
% Baseline，基线
Spectrum = Spectrum';
MoveMinima = zeros(length(Spectrum), 1);
for i = 1:length(Spectrum)
    if i < W/2
        MoveMinima(i, 1) = min( Spectrum(1:W) );
    elseif i >= W/2 && i <= ( length(Spectrum) - W/2 )
        MoveMinima(i, 1) = min( Spectrum(i-W/2+1:i+W/2) );
    elseif i > ( length(Spectrum) - W/2 )
        MoveMinima(i, 1) = min( Spectrum( (length(Spectrum) - W/2):end ) );
    end
end
%Baseline = smooth(Baseline, 30);
Baseline = zeros(length(Spectrum), 1);
for i = 1:length(Spectrum)
    if i < W/2
        Baseline(i, 1) = min( Spectrum(1:W) );
    elseif i >= W/2 && i <= ( length(Spectrum) - W/2 )
        Base = 0;
        for j =i-W/2+1:i+W/2
            Base = Base + MoveMinima(j);
        end
    Baseline(i, 1) = Base / W;
    elseif i > ( length(Spectrum) - W/2 )
        Baseline(i, 1) = min( Spectrum( (length(Spectrum) - W/2):end ) );
    end
    
end

SubBG_Spectrum = Spectrum - Baseline;
SubBG_Spectrum = SubBG_Spectrum';
Baseline = Baseline';