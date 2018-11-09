close all;
clear all;
clc;
%% Krishnan Schuler  Almeida  chenri Chakrabart our 
% ctrl+r ����ע��  ctrl+t����ȡ��
%13 15 17 21 23 23  7
%% psnr
% ker01 = [22.39   22.72  22.97  17.84  22.54 24.36];
% ker02 = [21.80   24.15  20.95  16.60  21.13 24.83];
% ker03 = [22.67   22.35  21.02  17.52  22.30 24.54]; 
% ker04 = [20.13   18.97  17.92  15.34  18.29 18.76];
% ker05 = [22.85   24.61  22.35  21.52  23.36 25.02];
% ker06 = [20.28   17.58  17.04  16.93  18.17 21.79];
% ker07 = [19.61   19.04  18.65  17.00  18.96 18.90];
% ker08 = [20.41   19.70  19.04  15.60  20.47 19.51];
%% isnr
% ker01 = [0.79   2.84  3.91  2.14   0.44 4.01];
% ker02 = [-0.04  3.41  2.01 -1.68   2.13 4.08];
% ker03 = [2.39   4.23  1.96 -0.09   3.05 4.73];
% ker04 = [-1.39  1.86  2.13  -1.36  2.08 0.93];
% ker05 = [2.91   4.35  1.26  4.38   2.74 4.73];
% ker06 = [0.32   0.22 -0.60  1.09   0.44 1.67];
% ker07 = [0.60   3.55  1.37  -1.30  2.19 1.84];
% ker08 = [-1.08  3.06  1.40  -2.18  3.50 1.39];
%% ssim
ker01 = [0.71    0.74 0.65  0.59   0.64 0.79];
ker02 = [0.65    0.75  0.74  0.47   0.71 0.74];
ker03 = [0.76    0.75  0.62  0.56   0.69 0.79];
ker04 = [0.53    0.52  0.41  0.30   0.50 0.36];
ker05 = [0.76    0.84  0.71  0.66   0.74 0.89];
ker06 = [0.56    0.42  0.35  0.40   0.43 0.56];
ker07 = [0.51    0.52  0.45  0.41   0.51 0.43];
ker08 = [0.54    0.57  0.49  0.36   0.62 0.47];
%% time 31 learn  Almeida chen neural our
% ker01 = [59.62   28.03   1192.63 116.32 253.85 0.3745];
% ker02 = [56.94   26.80   970.48  107.01 246.75 0.3915];
% ker03 = [28.99   27.21   956.39  115.55 239.44 0.3740];
% ker04 = [105.63  27.62   1376.04 107.08 282.79 0.6627];
% ker05 = [18.75   29.58   916.31  119.82  228.04 0.3724];
% ker06= [104.21   27.18   1198.36 111.46 240.10 0.3749];
% ker07 = [103.79  32.12   1319.47 119.42 279.90 0.3741];
% ker08 = [101.90  28.04   1206.74 115.55 299.98 0.3767];


%average = [im01; im02; im03; im04;im05;im06;im07;im08;im09;im10];
average = [ker01; ker02; ker03; ker04;ker05;ker06;ker07;ker08];
figure('color','white');bar(average,0.5);
ylabel('Average SSIM', 'FontName','Times New Roman','FontSize',10.5)
h = legend('Krishnan {\itet al}.',  'Schuler {\itet al}','Almeida {\itet al}','Chen {\itet al}','Chakrabarti {\itet al}',  'Ours', 1);
set(h, 'Fontname', 'Times New Roman', 'Fontsize', 10.5, 'Location',  'Southeast')
%set(gca, 'xticklabel', {'im01','im02','im03','im04','im05','im06','im07','im08','im09','im10'}, 'Fontname', 'Times New Roman', 'Fontsize', 10) ;
set(gca, 'xticklabel', {'kernel01','kernel02','kernel03','kernel04','kernel05','kernel06','kernel07','kernel08'}, 'Fontname', 'Times New Roman', 'Fontsize', 10) ;
axis([0.5 8.5 0 1]);
