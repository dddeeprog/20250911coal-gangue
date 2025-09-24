function [Data,Wavelength] = SifRead(pathname)
%读取sif文件（光谱和图像）
%{
输入：
pathname：读取文件的路径

输出：
Data：读出的数据
Wavelength:波长信息（读取图像时无）

日期：
20210616-ZD
%}
rc=atsif_setfileaccessmode(0);
rc=atsif_readfromfile(pathname);
if (rc == 22002)
    signal=0;
    [rc,present]=atsif_isdatasourcepresent(signal);
    if present
        [rc,no_frames]=atsif_getnumberframes(signal);
        for ii = 1:no_frames
            [rc,size]=atsif_getframesize(signal);
            [rc,left,bottom,right,top,hBin,vBin]=atsif_getsubimageinfo(signal,0);
            xaxis=0;
            [rc,data(:,ii)]=atsif_getframe(signal,ii-1,size);
            [rc,pattern]=atsif_getpropertyvalue(signal,'ReadPattern');
            if(pattern == '0')
                calibvals = zeros(1,size);
                for i=1:size,[rc,calibvals(i)]=atsif_getpixelcalibration(signal,xaxis,(i));
                end
                Data(:,ii) = data(:,ii);
                Wavelength = calibvals';
            elseif(pattern == '4')
                width = ((right - left)+1)/hBin;
                height = ((top-bottom)+1)/vBin;
                Data(:,:,ii)=reshape(data(:,ii),width,height);
                Wavelength = [];
            end
        end
    end
    atsif_closefile;
else
    disp('Could not load file.  ERROR - ');
    disp(rc);
end