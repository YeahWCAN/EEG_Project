clear all
filePath = strcat('D:\experiments\mycode\scaled_rhythmdata\original.mat');
datFile = load(filePath);
data = datFile.beta;

trialNum = 1280;

channelNum = 32;
fs = 128;
featureNum = 3;
windowtime = 5;
Nonlinear_feature = zeros(trialNum,featureNum,channelNum);
% tic;
for trialNo=1:trialNum
    disp(strcat('Extracting:  trialNo-',num2str(trialNo)));        
    for channelNo = 1:channelNum 
        signal = data(trialNo,channelNo,:);
        signal = squeeze(signal);
        NonlinearF = F_allNonlinearFeatures(128,signal,windowtime,0);
        Nonlinear_feature(trialNo,:,channelNo) = NonlinearF;         
    end                    
end
% toc;


C0_average= Nonlinear_feature(:,1,:);
M_H= Nonlinear_feature(:,2,:);
Average_PSen= Nonlinear_feature(:,3,:);

C0_average = squeeze(C0_average);
M_H = squeeze(M_H);
Average_PSen = squeeze(Average_PSen);

save C0_average C0_average
save M_H M_H
save Average_PSen Average_PSen
