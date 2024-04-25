% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
% This script provides the results in Table 1 concerning TKM.
addpath('./src/ALS')
addpath('./src/features')
addpath('./src/utils')
close all
warning('off','all');
%% Airline Dataset
M = 64;
P = 10;
NTrials = 10;
maxIte = 25;
NSet = 5929413;
RSet = [4,6,8];
lambda = 1e-10;
train = zeros(NTrials,numel(NSet),numel(RSet));
test = zeros(NTrials,numel(NSet),numel(RSet));
wallTime = zeros(NTrials,numel(NSet),numel(RSet));
warning('off','all');
NIdx = 0;
for N = NSet
    NIdx = NIdx+1;
    RIdx = 0;
    features = @(X) FourierFeaturesFull(X,M,P);
    for R = RSet
        RIdx = RIdx+1;
        for ite = 1:NTrials
            rng(ite);
            X = readmatrix('datasetsStorage/airline.csv');  % data already preprocessed
            perm = randperm(size(X,1));
            X = X(perm,:);
            X = X(1:N,:);
            X = X(1:floor(2*N/3),:);    %train on 2/3 of the data
            Y = X(:,end);
            X = X(:,1:end-1);
          
            YMean = mean(Y);    YStd = std(Y);
            XMin = min(X);  XMax = max(X);
            Y = (Y-YMean)./YStd;
            X = (X-XMin)./(XMax-XMin);
            
            % Train
            disp("N: "+string(N)+" R: "+string(R)+" ite: "+string(ite));
            tic;tic;
            [WCP,loss,~] = CPLSFull(X,Y,features, M,R,lambda,maxIte);
            wallTime(ite,NIdx,RIdx) = toc;toc;
            train(ite,NIdx,RIdx) = mean((Y-CPPredictFull(X,features,WCP)).^2);

            % Test
            clear X Y
            X = readmatrix('datasetsStorage/airline.csv');
            X = X(perm,:);
            X = X(1:N,:);
            X = X(floor(2*N/3)+1:end,:);    %test on 1/3 of the data
            Y = X(:,end);
            X = X(:,1:end-1);
            X = (X-XMin)./(XMax-XMin);
            Y = (Y-YMean)./YStd;
            test(ite,NIdx,RIdx) = mean((Y-CPPredictFull(X,features,WCP)).^2);
            disp('Test error: '+string(test(ite,NIdx,RIdx)));
            save("airline_TKM"+"M"+string(M)+"P"+string(P)+"RMax"+string(max(RSet))+"maxIte"+string(maxIte)+".mat",'train','test','wallTime','loss','RSet','P','M');
        end
    end
end