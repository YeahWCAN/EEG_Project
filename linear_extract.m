clear all
filePath = strcat('D:\experiments\mycode\scaled_rhythmdata\original.mat');
datFile = load(filePath);
data = datFile.beta;

trialNum = 1280;
channelNum = 32;
fs = 128;
featureNum = 6;
windowtime = 5;
linear_feature = zeros(trialNum,featureNum,channelNum);
for trialNo=1:trialNum
    disp(strcat('Extracting:  trialNo-',num2str(trialNo)));        
    for channelNo = 1:channelNum 
        signal = data(trialNo,channelNo,:);
        signal = squeeze(signal);
        linearF = F_allLinearFeatures(128,signal,windowtime,0);
        linear_feature(trialNo,:,channelNo) = linearF;         
    end                    
end


PPmean = linear_feature(:,1,:);
meanSquare= linear_feature(:,2,:);
var= linear_feature(:,3,:);
maxf= linear_feature(:,4,:);
maxpsd= linear_feature(:,5,:);
sumPower= linear_feature(:,6,:);

PPmean = squeeze(PPmean);
meanSquare = squeeze(meanSquare);
var = squeeze(var);
maxf = squeeze(maxf);
maxpsd = squeeze(maxpsd);
sumPower = squeeze(sumPower);

save PPmean PPmean
save meanSquare meanSquare
save var var
save maxf maxf
save maxpsd maxpsd
save sumPower sumPower
