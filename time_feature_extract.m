filePath = strcat('D:\experiments\mycode\scaled_rhythmdata\original.mat');
datFile = load(filePath);
data = datFile.beta;

trialNum = 1280;
channelNum = 32;
fs = 128;

Mean = zeros(1280,32);
Var = zeros(1280,32);
Zcr = zeros(1280,32);
Kur = zeros(1280,32);
Ske = zeros(1280,32);
Shan = zeros(1280,32);
for trialNo = 1:1280
    disp(strcat('Extracting:  trialNo-',num2str(trialNo)));
    for channelNo = 1:channelNum
        %disp(strcat('entropy Extracting:  trialNo-',num2str(trialNo)));
        me = zeros(1,10);
        va =  zeros(1,10);
        zc =  zeros(1,10);
        ku =  zeros(1,10);
        sk =  zeros(1,10);
        shan=  zeros(1,10);
        for t = 1:10
            start_point = (t-1)*fs*5+1;
            end_point = t*fs*5;
            signal = data(trialNo,channelNo,start_point:end_point);
            signal = squeeze(signal);
            me(t) = mean(signal);
            va(t) = var(signal);
            zc(t) = zcr(signal);
            ku(t) = kurtosis(signal');
            sk(t) = skewness(signal');
            shan(t) = wentropy(signal,'shannon');
        end
        Mean(trialNo,channelNo) = mean(me);
        Var(trialNo,channelNo) = mean(va);
        Zcr(trialNo,channelNo) = mean(zc);
        Kur(trialNo,channelNo) = mean(ku);
        Ske(trialNo,channelNo) = mean(sk);
        Shan (trialNo,channelNo) = mean(shan);   
    end
end
save Mean Mean
save Var Var
save Zcr Zcr
save Kur Kur
save Ske Ske
save Shan Shan
        
