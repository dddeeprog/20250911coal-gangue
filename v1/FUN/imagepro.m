% [300 260 350 350]  第一次波动性实验  Ca(air)2
% [210  110 500 500]  第二次波动性实验
% [279 229 350 350] Ca(air)1                      %以上均为202211月实验




%% read         
clear
clc
% [ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image', 'MultiSelect', 'on');
File = {'1'             %sif文件名
        '2'
%         'Ca_3'
%         'Ca_4'
%         'Ca_5'
%         'Ca_6'
%         'Ca_7'
%         'Ca_8'
%         'Ca_9'
        };
Num = size(File, 1);      %sif文件个数
figure(1);
for i = 1:Num             %依次读取sif文件
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);     %文件名，不加后缀
 
    Mat = [];
    for j = 1:15                  %sif文件中的前j个图像
        Img = data(:, :, j);       
        Img = imrotate(Img, 90);  
%         Img2 = imcrop(Img,  [280 350 350 350]);
          Img2 = imcrop(Img,  [300 260 350 350]);   %选取像素区域
        Mat = [Mat; Img2];
    end
    h = imshow(Mat, [], 'border','tight');
    
    colormap(jet)
    axis image
    axis off
    saveas(gcf, [delay '.jpg'])
end

% colorbar(axes1);

%% merge
ImgMat = [];
for i = 1:Num
    ImgFile = [File{i} '.jpg'];
    Img = imread(ImgFile);
    ImgMat = [ImgMat, Img];
end
imwrite(ImgMat, 'PlasmaImg.jpg');

%% Average Image
clear
clc
% [ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image', 'MultiSelect', 'on');
File = {
        'Na_1'
        'Na_2'
        'Na_3'
        'Na_4'
        'Na_5'
        'Na_6'
        'Na_7'
        'Na_8'
        'Na_9'
        'Na_10'
        'Na_11'
        'Na_12'
        };
Num = size(File, 1);
figure(1);

%平均后的图像，不做归一化
for i = 1:Num
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
       Mat = imcrop(Mat,  [210  110 500 500]); %前两个数为图像左下角坐标，后两位数是矩形边长
 
    h = imshow(Mat, [], 'border','tight');
    colormap(jet)
     caxis([300,10000]); %Ca    %选择colorbar范围
%       caxis([500,10000]); %Na  
   
    axis image
    axis off
    saveas(gcf, [delay '_Ave.jpg']);
end

%平均后的图像，做归一化
for i = 1:Num
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
      Mat = imcrop(Mat,  [210  110 500 500]);
      Mat = Mat./(max(Mat(:)))*100;
 
    h = imshow(Mat, [], 'border','tight');
    colormap(jet)
    axis image
    axis off
    saveas(gcf, [delay '_Ave.jpg'])
end

% 画colorbar
figure(2);
c = colorbar;
colormap(jet)
 caxis([300,3200]);
c.Ticks = [300,3200];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, ['colorbar.jpg'])


% 画归一化colorbar
figure(2);
c = colorbar;
colormap(jet)
 caxis([0,100]);
c.Ticks = [0 100];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, ['colorbar.jpg'])

% merge

%图像排成一行或是一列
%  ImgMat = [];
% for i = 1:Num     
%     ImgFile = [File{i} '_Ave.jpg'];
%     Img = imread(ImgFile);
%     ImgMat = [ImgMat, Img];
% end

%图像分行
ImgMat1 = [];
ImgMat2 = [];

for i = 1:6
    ImgFile = [File{i} '_Ave.jpg'];
    Img = imread(ImgFile);
    ImgMat1 = [ImgMat1, Img];
end

 for i = 7:12
% for i = [7 8 13 14 15 16]
    ImgFile = [File{i} '_Ave.jpg'];
    Img = imread(ImgFile);
    ImgMat2 = [ImgMat2, Img];
   
end

%   caxis([500,3600]);
%   colorbar;
imwrite(ImgMat1, 'PlasmaImg_Ave1.jpg');
imwrite(ImgMat2, 'PlasmaImg_Ave2.jpg');

%% RSD Image
clear
clc
 File = { 'Na_1'
        'Na_2'
        'Na_3'
        'Na_4'
        'Na_5'
        'Na_6'
        'Na_7'
        'Na_8'
        'Na_9'
        'Na_10'
        'Na_11'
        'Na_12'

%      'noFilter_1'
%         'noFilter_2'
%         'noFilter_3'
%         'noFilter_4'
%         'noFilter_5'
%         'noFilter_6'
%         'noFilter_7'
%         'noFilter_8'
%         'noFilter_9'
%     'Na_1'
%         'Na_2'
%         'Na_3'
%         'Na_4'
%         'Na_5'
%         'Na_6'
%         'Na_7'
%         'Na_8'
%         'Na_9'
        };
Num = size(File, 1);
figure(1);
for i = 1:Num
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
%      data = data(:,:,1:20);  %前20幅
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(501, 501, 50);
    for j = 1:50
%         TarMat(:, :, j) = imcrop(data2(:, :, j),  [280 350 350 350]);
TarMat(:, :, j) = imcrop(data2(:, :, j),  [210  110 500 500]);
    end
    RSDMat = zeros(501);
    for j = 1:501
        for k = 1:501
            arr = TarMat(j, k, :);
            Ave = mean(arr);
            Std = std(arr);
            RSDMat(j, k) = Std / Ave;
        end
    end
    h = imshow(RSDMat, [], 'border','tight');
    colormap(jet)
    caxis([0 0.3]);
%     axis image
%     axis off
%     saveas(gcf, [delay '_Std.jpg'])
end

% merge
% ImgMat = [];
% for i = 1:Num
%     ImgFile = [File{i} '_Std.jpg'];
%     Img = imread(ImgFile);
%     ImgMat = [ImgMat, Img];
% end
ImgMat1 = [];
ImgMat2 = [];
for i = 1:6
    ImgFile = [File{i} '_Std.jpg'];
    Img = imread(ImgFile);
    ImgMat1 = [ImgMat1, Img];
end

for i = 7:12
%     for i = [7 8 13 14 15 16]
    ImgFile = [File{i} '_Std.jpg'];
    Img = imread(ImgFile);
    ImgMat2 = [ImgMat2, Img];
end

imwrite(ImgMat1, 'PlasmaImg_Std1.jpg');
imwrite(ImgMat2, 'PlasmaImg_Std2.jpg');


% 画colorbar
figure(2);
c = colorbar;
colormap(jet)
 caxis([0,0.3]);
c.Ticks = [0 0.3];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, [ 'colorbar.jpg'])

%%
%提取RSD图像
clear
clc
 File = { 'Na_1'
%         'Na_2'
%         'Na_3'
%         'Na_4'
%         'Na_5'
%         'Na_6'
%         'Na_7'
%         'Na_8'
%         'Na_9'
%         'Na_10'
%         'Na_11'
%         'Na_12'

%      'noFilter_1'
%         'noFilter_2'
%         'noFilter_3'
%         'noFilter_4'
%         'noFilter_5'
%         'noFilter_6'
%         'noFilter_7'
%         'noFilter_8'
%         'noFilter_9'

        };
Num = size(File, 1);
figure(1);
RSDHist = zeros(Num,501*501);
                                                                                                                                                                                                                                                                                                                                                                                                                          
for i = 1:Num
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
%   data = data(:,:,1:20);  %前20幅
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(501, 501, 50); %截取的50幅图像
    RSDMat = zeros(501);          %RSD图
    RSDMat2 = zeros(1,501*501); 
    RSDMat3 = zeros(1,501*501); %RSD数值提取


            for j = 1:50
%              TarMat(:, :, j) = imcrop(data2(:, :, j),  [280 350 350 350]);
               TarMat(:, :, j) = imcrop(data2(:, :, j),   [210  110 500 500]);
            end
   
            for j = 1:501
                 for k = 1:501
                    arr = TarMat(j, k, :);
                    Ave = mean(arr);
                    Std = std(arr);
                    RSDMat(j, k) = Std / Ave;
                 end
            end
    RSDMat(find(RSDMat<0.04))  = 0;
    h = imshow(RSDMat, [], 'border','tight');
    colormap(jet);
    caxis([0 0.3]);
    saveas(gcf, [delay '_Std_extract.jpg']);

end

ImgMat1 = [];
ImgMat2 = [];
for i = 1:6
    ImgFile = [File{i} '_Std_extract.jpg'];
    Img = imread(ImgFile);
    ImgMat1 = [ImgMat1, Img];
end

for i = 7:12
%     for i = [7 8 13 14 15 16]
    ImgFile = [File{i} '_Std_extract.jpg'];
    Img = imread(ImgFile);
    ImgMat2 = [ImgMat2, Img];
end

imwrite(ImgMat1, 'PlasmaImg_Std1_extract.jpg');
imwrite(ImgMat2, 'PlasmaImg_Std2_extract.jpg');

% 画colorbar
figure(2);
c = colorbar;
colormap(jet)
 caxis([0,0.3]);
c.Ticks = [0 0.3];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, [ 'colorbar.jpg']);

%%
%直方图分布
clear
clc
 File = { 'Na_1'
        'Na_2'
        'Na_3'
        'Na_4'
        'Na_5'
        'Na_6'
        'Na_7'
        'Na_8'
        'Na_9'
        'Na_10'
        'Na_11'
        'Na_12'

%      'noFilter_1'
%         'noFilter_2'
%         'noFilter_3'
%         'noFilter_4'
%         'noFilter_5'
%         'noFilter_6'
%         'noFilter_7'
%         'noFilter_8'
%         'noFilter_9'

        };
Num = size(File, 1);
figure(1);
RSDHist = zeros(Num,501*501);
RSDMat4 = [];
                                                                                                                                                                                                                                                                                                                                                                                                                          
for i = 1:Num
    ImgFile = [File{i} '.sif'];
    data = sifreadnk_img(ImgFile);
%   data = data(:,:,1:20);  %前20幅
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(501, 501, 50); %截取的50幅图像
    RSDMat = zeros(501);          %RSD图
    RSDMat2 = zeros(1,501*501); 
    RSDMat3 = zeros(1,501*501); %RSD数值提取


            for j = 1:50
%              TarMat(:, :, j) = imcrop(data2(:, :, j),  [280 350 350 350]);
               TarMat(:, :, j) = imcrop(data2(:, :, j),   [210  110 500 500]);
            end
   
            for j = 1:501
                 for k = 1:501
                    arr = TarMat(j, k, :);
                    Ave = mean(arr);
                    Std = std(arr);
                    RSDMat(j, k) = Std / Ave;
                 end
            end
    h = imshow(RSDMat, [], 'border','tight');
    colormap(jet);
    caxis([0 0.3]);
%     saveas(gcf, [delay '_Std.jpg']);

    RSDMat2 = reshape(RSDMat,[1,501*501]);
    RSDHist(i,:) = sort(RSDMat2,2);
end

RSDHist = RSDHist';
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

%% 图像像素（强度）求和

%等离子体区域不做提取
clear
clc
File = { 'air_200mj_delay20us_width140us_Gain2000__4Hz_Na591Filter'
%         'Ca_2'
%         'Ca_3'
%         'Ca_4'
%         'Ca_5'
%         'Ca_6'
%         'Ca_7'
%         'Ca_8'
%         'Ca_9'
    };
Num = size(File, 1);

result = zeros(Num,3); %图像定量结果：强度平均值，强度std，RSD
    for i = 1:Num
        ImgFile = [File{i} '.sif'];
        data = sifreadnk_img(ImgFile);
        delay = ImgFile(1:end-4);
        data2 = imrotate(data, 90);
        TarMat = zeros(501, 501, 50);
             for j = 1:50
                TarMat(:, :, j) = imcrop(data2(:, :, j),  [[210  110 500 500]]) ;
             end
        IntSumMat = zeros(50,1);
        S1 = sum(TarMat,[1 2]);   %单幅图像强度总和
              for j = 1:50
                  IntSumMat(j,:) = S1(:,:,j);
              end
              Ave = mean(IntSumMat);
              Std = std(IntSumMat);
              RSD = Std/Ave*100;
    
              result(i,1) = Ave;
              result(i,2) = Std;
              result(i,3) = RSD;
    end


%气溶胶等离子体扣除空气等离子体平均图像
clear
clc
% [ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image', 'MultiSelect', 'on');
File1 = {                    %含气溶胶等离子体图像
        '5ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '15ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '20ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '25ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '50ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '100ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '200ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '400ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '600ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '800ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        
        
       
        };
File2 = {                   %空气等离子体图像
   'air_200mj_delay2us_width9us_Gain2000_4Hz_Ca393Filter'
%         'Na_591_air_1'
%         'Na_591_air_2'
%         'Na_591_air_3'
%         'Na_591_air_4'
%         'Na_591_air_5'
%         'Na_591_air_6'
%         'Na_591_air_7'
%         'Na_591_air_8'
%         'Na_591_air_9'
%         'Na_591_air_10'
%         'Na_591_air_11'
%         'Na_591_air_12'
%         'Ca_393_air_3'
%         'Ca_393_air_4'
%         'Ca_393_air_5'
%         'Ca_393_air_6'
%         'Ca_393_air_7'
%         'Ca_393_air_8'
%         'Ca_393_air_9'
      
        };

Num1 = size(File1, 1);
Num2 = size(File2, 1);
result = zeros(Num1,3); %图像定量结果：强度平均值，强度std，RSD
pixl = 350;
window = pixl+1;

figure(1);
Airmat = zeros(window,window,Num2);
for i = 1:Num2
    ImgFile = [File2{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
    Mat = imcrop(Mat,  [300 260 350 350] ); %前两个数为图像左下角坐标，后两位数是矩形边长
    Airmat(:,:,i) = Mat;

    h = imshow(Mat, [], 'border','tight');
    colormap(jet);
    caxis([500,1000]); %Ca    %选择colorbar范围
   
    axis image
    axis off
    saveas(gcf, [delay '_Ave_air.jpg']);
end

MinPixls  = zeros(Num1,1);  %扣背景后，所有图像中的最小值像素
for i = 1:Num1
    ImgFile = [File1{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(window, window, 50);
         for j = 1:50
            TarMat(:, :, j) = imcrop(data2(:, :, j),  [[300 260 350 350] ]);
%             TarMat(:, :, j) = TarMat(:, :, j) -  Airmat(:,:,i) ;
            TarMat(:, :, j) = TarMat(:, :, j) -  Airmat  ;
         end

    MinPixls(i,:) = min(TarMat(:));

    IntSumMat = zeros(50,1);
    S1 = sum(TarMat,[1 2]);   %单幅图像强度总和
          for j = 1:50
              IntSumMat(j,:) = S1(:,:,j);
          end
          Ave = mean(IntSumMat);
          Std = std(IntSumMat);
          RSD = Std/Ave*100;

          result(i,1) = Ave;
          result(i,2) = Std;                             
          result(i,3) = RSD;
end

%SNR  空气等离子体图像强度和的std作为噪声（先每幅图像强度求和，再求std）
for i = 1:Num2
    ImgFile = [File2{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    AirMat2 = zeros(window, window, 50);
         for j = 1:50
            AirMat2(:, :, j) = imcrop(data2(:, :, j),  [[300 260 350 350] ]);
         end
    IntSumMat2 = zeros(50,1);
    S2 = sum(AirMat2,[1 2]);   %单幅图像强度总和
          for j = 1:50
              IntSumMat2(j,:) = S2(:,:,j);
          end
          Ave2 = mean(IntSumMat2);
          Std2 = std(IntSumMat2);
          RSD2 = Std2/Ave2*100;

          result2(i,1) = Ave2;
          result2(i,2) = Std2;
          result2(i,3) = RSD2;
end

%求SNR

%     SNR = result(:,1)./result2(:,2);
     SNR = result(:,1)./result2(:,2);

%% 图像作差(平均-平均)
clear
clc
% [ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image', 'MultiSelect', 'on');
File1 = {                    %含气溶胶等离子体图像
        'Na_1'
        'Na_2'
        'Na_3'
        'Na_4'
        'Na_5'
        'Na_6'
        'Na_7'
        'Na_8'
        'Na_9'
        'Na_10'
        'Na_11'
        'Na_12'
        };
File2 = {                   %空气等离子体图像
        'Na_591_air_1'
        'Na_591_air_2'
        'Na_591_air_3'
        'Na_591_air_4'
        'Na_591_air_5'
        'Na_591_air_6'
        'Na_591_air_7'
        'Na_591_air_8'
        'Na_591_air_9'
        'Na_591_air_10'
        'Na_591_air_11'
        'Na_591_air_12'
        };
Num = size(File2, 1);

%先得出空气等离子体的平均图像
%平均后的图像，不做归一化
figure(1);
Airmat = zeros(501,501,Num);
for i = 1:Num
    ImgFile = [File2{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
    Mat = imcrop(Mat,  [210  110 500 500]); %前两个数为图像左下角坐标，后两位数是矩形边长
    Airmat(:,:,i) = Mat;

    h = imshow(Mat, [], 'border','tight');
    colormap(jet);
    caxis([300,10000]); %Ca    %选择colorbar范围
   
    axis image
    axis off
%     saveas(gcf, [delay '_Ave_air.jpg']);
end

%再将含气溶胶图像做平均
figure(2);
for i = 1:Num
    ImgFile = [File1{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
    Mat = imcrop(Mat,  [210  110 500 500]); %前两个数为图像左下角坐标，后两位数是矩形边长
    AerosolMat(:,:,i) = Mat;

    h = imshow(Mat, [], 'border','tight');
    colormap(jet);
    caxis([300,10000]); %Ca    %选择colorbar范围
   
    axis image
    axis off
%     saveas(gcf, [delay '_Ave_plasma.jpg']);
end

    
%未做归一化（图像）
    figure(3);
    ExtractMat = zeros(501,501,Num);
    for i = 1:Num
        ExtractMat = AerosolMat(:,:,i) - Airmat(:,:,i);
    
        h = imshow(ExtractMat, [], 'border','tight');
        colormap(jet);
         caxis([0,3000]); %Ca    %选择colorbar范围
       
        axis image
        axis off
        saveas(gcf, [num2str(i) '_Ave_Extract.jpg']);
    end

    ImgMat1 = [];
    ImgMat2 = [];
        for i = 1:6
            ImgFile = [num2str(i) '_Ave_Extract.jpg'];
            Img = imread(ImgFile);
            ImgMat1 = [ImgMat1, Img];
        end

        for i = 7:12
%     for i = [7 8 13 14 15 16]
            ImgFile = [num2str(i) '_Ave_Extract.jpg'];
            Img = imread(ImgFile);
            ImgMat2 = [ImgMat2, Img];
        end

  imwrite(ImgMat1, 'PlasmaImg_Ave_extract1.jpg');
    imwrite(ImgMat2, 'PlasmaImg_Ave_extract2.jpg');

    % 画colorbar
        figure(4);
        c = colorbar;
        colormap(jet)
         caxis([0,3000]);
        c.Ticks = [0,3000];
        c.FontWeight = 'bold';
        c.FontSize = 30;
        c.Position(1) = 0.1 ;
        c.Position(3) = 0.06 ;
        c.Position(4) = 0.8 ;
        c.AxisLocation = 'in';
        axis image
        axis off
        saveas(gcf, ['colorbar.jpg'])

%归一化（图像）
    figure(3);
    ExtractMat = zeros(501,501,Num);
    for i = 1:Num
    ExtractMat = AerosolMat(:,:,i) - Airmat(:,:,i);
    ExtractMat = ExtractMat./(max(ExtractMat(:)))*100;

    h = imshow(ExtractMat, [], 'border','tight');
    colormap(jet);
    caxis([0,100]); %Ca    %选择colorbar范围
   
    axis image
    axis off
    saveas(gcf, [num2str(i) '_Ave_Extract.jpg']);
    end

    ImgMat1 = [];
    ImgMat2 = [];
        for i = 1:6
            ImgFile = [num2str(i) '_Ave_Extract.jpg'];
            Img = imread(ImgFile);
            ImgMat1 = [ImgMat1, Img];
        end

        for i = 7:12
%     for i = [7 8 13 14 15 16]
            ImgFile = [num2str(i) '_Ave_Extract.jpg'];
            Img = imread(ImgFile);
            ImgMat2 = [ImgMat2, Img];
        end

    imwrite(ImgMat1, 'PlasmaImg_Ave_extract1.jpg');
    imwrite(ImgMat2, 'PlasmaImg_Ave_extract2.jpg');


    % 画归一化colorbar
        figure(4);
        c = colorbar;
        colormap(jet)
         caxis([0,100]);
        c.Ticks = [0 100];
        c.FontWeight = 'bold';
        c.FontSize = 30;
        c.Position(1) = 0.1 ;
        c.Position(3) = 0.06 ;
        c.Position(4) = 0.8 ;
        c.AxisLocation = 'in';
        axis image
        axis off
        saveas(gcf, ['colorbar.jpg']);

 %% 图像作差(单幅-平均)     平均强度应该一致，但可以计算扣除后图像的波动性。

clear
clc
% [ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image', 'MultiSelect', 'on');
File1 = {                    %含气溶胶等离子体图像
        '5ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '15ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '20ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '25ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '50ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '100ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '200ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '400ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '600ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        '800ppm_200mj_delay2us_width9us_Gain2000_4Hz'
        };
File2 = {                   %空气等离子体图像
   'air_200mj_delay2us_width9us_Gain2000_4Hz_Ca393Filter'
%         'Na_591_air_1'
%         'Na_591_air_2'
%         'Na_591_air_3'
%         'Na_591_air_4'
%         'Na_591_air_5'
%         'Na_591_air_6'
%         'Na_591_air_7'
%         'Na_591_air_8'
%         'Na_591_air_9'
%         'Na_591_air_10'
%         'Na_591_air_11'
%         'Na_591_air_12'
%         'Ca_393_air_3'
%         'Ca_393_air_4'
%         'Ca_393_air_5'
%         'Ca_393_air_6'
%         'Ca_393_air_7'
%         'Ca_393_air_8'
%         'Ca_393_air_9'
      
        };
Num1 = size(File1, 1);
Num2 = size(File2, 1);
pixl = 350;
window = pixl+1;

%先得出空气等离子体的平均图像
%平均后的图像，不做归一化
figure(1);
Airmat = zeros(window,window,Num2);
for i = 1:Num2
    ImgFile = [File2{i} '.sif'];
    data = sifreadnk_img(ImgFile);
    delay = ImgFile(1:end-4);
    
    Mat = mean(data, 3);         % 像素点平均
    Mat = imrotate(Mat, 90);
%     Mat = imcrop(Mat,  [280 350 350 350]);
    Mat = imcrop(Mat,  [300 260 350 350]  ); %前两个数为图像左下角坐标，后两位数是矩形边长
    Airmat(:,:,i) = Mat;

    h = imshow(Mat, [], 'border','tight');
    colormap(jet);
    caxis([300,10000]); %Ca    %选择colorbar范围
   
    axis image
    axis off
    saveas(gcf, [delay '_Ave_air.jpg']);
end

%再将单幅含气溶胶图像扣除空气等离子体平均图像
figure(2);
AveMat = zeros(window,window,Num1);
RSDmat = zeros(window,window,Num1);
MinTarMat = zeros(Num1,1);

 MinPixls  = zeros(Num1,1);  %扣背景后，所有图像中的最小值像素
for i = 1:Num1
    ImgFile = [File1{i} '.sif'];
    data = sifreadnk_img(ImgFile);
%      data = data(:,:,1:20);  %前20幅
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(window, window, 50);
    for j = 1:50
%         TarMat(:, :, j) = imcrop(data2(:, :, j),  [280 350 350 350]);
        TarMat(:, :, j) = imcrop(data2(:, :, j),  [300 260 350 350]  );
        TarMat(:,:,j) = TarMat(:, :, j) - Airmat(:,:,i);
%         TarMat(:,:,j) = TarMat(:, :, j) - Airmat;
           TarMat(:,:,j) = TarMat(:,:,j)+600;
    end
     
    MinPixls(i,:) = min(TarMat(:));


    AveMat(:,:,i) = mean(TarMat,3);

    RSDMat = zeros(window);
    for j = 1:window
        for k = 1:window
            arr = TarMat(j, k, :);
            Ave = mean(arr);
            Std = std(arr);
            RSDMat(j, k) = Std / Ave;
        end
    end
    
    RSDmat(:,:,i) = RSDMat;

    h = imshow(RSDMat, [], 'border','tight');
    colormap(jet)
    caxis([0 0.3]);
%     axis image
%     axis off
    saveas(gcf, [num2str(i) '_Std.jpg'])
end

% 画colorbar
figure(3);
c = colorbar;
colormap(jet)
 caxis([0,0.5]);
c.Ticks = [0 0.5];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, [ 'colorbar1.jpg'])

for i = 7:Num
    ImgFile = [File1{i} '.sif'];
    data = sifreadnk_img(ImgFile);
%      data = data(:,:,1:20);  %前20幅
    delay = ImgFile(1:end-4);
    data2 = imrotate(data, 90);
    TarMat = zeros(501, 501, 50);
    for j = 1:50
%         TarMat(:, :, j) = imcrop(data2(:, :, j),  [280 350 350 350]);
        TarMat(:, :, j) = imcrop(data2(:, :, j),  [210  110 500 500]);
        TarMat(:,:,j) = TarMat(:, :, j) - Airmat(:,:,i);
        TarMat(:,:,j) = TarMat(:,:,j) +600;
    end
    AveMat(:,:,i) = mean(TarMat,3);

    RSDMat = zeros(501);
    for j = 1:501
        for k = 1:501
            arr = TarMat(j, k, :);
            Ave = mean(arr);
            Std = std(arr);
            RSDMat(j, k) = Std / Ave;
        end
    end
    h = imshow(RSDMat, [], 'border','tight');
    colormap(jet)
    caxis([0 0.2]);
%     axis image
%     axis off
    saveas(gcf, [num2str(i) '_Std.jpg'])
end

% 画colorbar
figure(3);
c = colorbar;
colormap(jet)
 caxis([0,0.2]);
c.Ticks = [0 0.2];
c.FontWeight = 'bold';
c.FontSize = 30;
c.Position(1) = 0.1 ;
c.Position(3) = 0.06 ;
c.Position(4) = 0.8 ;
c.AxisLocation = 'in';
axis image
axis off
saveas(gcf, [ 'colorbar2.jpg'])

% RSD图像
ImgMat1 = [];
ImgMat2 = [];
for i = 1:6
    ImgFile = [num2str(i) '_Std.jpg'];
    Img = imread(ImgFile);
    ImgMat1 = [ImgMat1, Img];
end

for i = 7:Num1
%     for i = [7 8 13 14 15 16]
    ImgFile = [num2str(i) '_Std.jpg'];
    Img = imread(ImgFile);
    ImgMat2 = [ImgMat2, Img];
end

imwrite(ImgMat1, 'PlasmaImg_Std1.jpg');
imwrite(ImgMat2, 'PlasmaImg_Std2.jpg');


%未归一化平均图像


figure(4);
for i = 1:Num1
        AveMatrix = AveMat(:,:,i);

        h = imshow(AveMatrix, [], 'border','tight');
        colormap(jet)
         caxis([300 2300]);
%     axis image
%     axis off
        saveas(gcf, [num2str(i) '_Ave.jpg'])
end

    ImgMat1 = [];
    ImgMat2 = [];
        for i = 1:5
            ImgFile = [num2str(i) '_Ave.jpg'];
            Img = imread(ImgFile);
            ImgMat1 = [ImgMat1, Img];
        end

        for i = 6:Num
%     for i = [7 8 13 14 15 16]
            ImgFile = [num2str(i) '_Ave.jpg'];
            Img = imread(ImgFile);
            ImgMat2 = [ImgMat2, Img];
        end

    imwrite(ImgMat1, 'PlasmaImg_Ave1.jpg');
    imwrite(ImgMat2, 'PlasmaImg_Ave2.jpg');

    % 画归一化colorbar
        figure(5);
        c = colorbar;
        colormap(jet)
         caxis([300,2300]);
        c.Ticks = [300 2300];
        c.FontWeight = 'bold';
        c.FontSize = 30;
        c.Position(1) = 0.1 ;
        c.Position(3) = 0.06 ;
        c.Position(4) = 0.8 ;
        c.AxisLocation = 'in';
        axis image
        axis off
        saveas(gcf, ['colorbar.jpg']);


%归一化平均图像


figure(4);
for i = 1:Num1
        AveMatrix = AveMat(:,:,i);
        AveMatrix = AveMatrix./(max(AveMatrix(:)))*100;

        h = imshow(AveMatrix, [], 'border','tight');
        colormap(jet)
%         caxis([0 0.3]);
%     axis image
%     axis off
        saveas(gcf, [num2str(i) '_Ave.jpg'])
end

    ImgMat1 = [];
    ImgMat2 = [];
        for i = 1:6
            ImgFile = [num2str(i) '_Ave.jpg'];
            Img = imread(ImgFile);
            ImgMat1 = [ImgMat1, Img];
        end

        for i = 7:Num1
%     for i = [7 8 13 14 15 16]
            ImgFile = [num2str(i) '_Ave.jpg'];
            Img = imread(ImgFile);
            ImgMat2 = [ImgMat2, Img];
        end

    imwrite(ImgMat1, 'PlasmaImg_Ave1.jpg');
    imwrite(ImgMat2, 'PlasmaImg_Ave2.jpg');

    % 画归一化colorbar
        figure(5);
        c = colorbar;
        colormap(jet)
         caxis([0,100]);
        c.Ticks = [0 100];
        c.FontWeight = 'bold';
        c.FontSize = 30;
        c.Position(1) = 0.1 ;
        c.Position(3) = 0.06 ;
        c.Position(4) = 0.8 ;
        c.AxisLocation = 'in';
        axis image
        axis off
        saveas(gcf, ['colorbar.jpg']);

%% read single image
[ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image');
data = sifreadnk_img([ImgPath, ImgFile] );
Img = data(:, :, 1);
Img = imrotate(Img, 90);
Img2 = imcrop(Img,[280 350 350 350]);    
imshow(Img2, [], 'border','tight');
colormap(jet)
axis image

%% 标尺
[ImgFile, ImgPath] = uigetfile('*.sif', 'Read Image');
Img = sifreadnk_img([ImgPath, ImgFile] );
Img = imrotate(Img, 90);
Img2 = imcrop(Img,[470 550 250 250]);    
imshow(Img2, [], 'border','tight');
colormap(jet)
axis image
saveas(gcf, 'Ruler.jpg')

