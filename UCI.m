addpath('./src/ALS')
addpath('./src/features')
addpath('./src/RFF')
addpath('./src/utils')
close all
warning('off','all');
%% Hyperparameters
MSet = 16;
PSet = 1+2.^(-2:6);
lambdaSet = 10.^(-5:5);
maxIteSet = 5000;
NTrials = 10;
%% Run
allFolders = dir("./datasets");
allFolders(1:2) = [];
for i = 1:length(allFolders)
    file = dir(fullfile(allFolders(i).name));
    filename = string([allFolders(i).name, file.name]);
    for M = MSet
        % Cross-Validation for lambda and P
        rng('default');
        X = readmatrix("./datasets/"+string(filename));
        perm = randperm(size(X,1));
        X = X(perm,:);
        XTest = X(floor(0.8*size(X,1))+1:end,:);
        X = X(1:floor(0.8*size(X,1)),:);
        Y = X(:,end);
        YTest = XTest(:,end);
        XTest = XTest(:,1:end-1);
        X = X(:,1:end-1);
        [N,~] = size(X);
        YMean = mean(Y);
        YStd = std(Y);
        XMin = min(X);
        XMax = max(X);
        Y = (Y-YMean)./YStd;
        X = (X-XMin)./(XMax-XMin)-0.5;
        YTest = (YTest-YMean)./YStd;
        XTest = (XTest-XMin)./(XMax-XMin)-0.5;

        % Determine lambda by 3-fold cv
        c = cvpartition(size(X,1),'KFold',3);
        valError = zeros(numel(lambdaSet),numel(PSet),3);
        lambdaIdx = 0;
        for lambda = lambdaSet
            PIdx = 0;
            lambdaIdx = lambdaIdx+1;
            for P = PSet
                PIdx = PIdx+1;
                featuresFull = @(X) FourierFeaturesFull(X,M,P); 
                for fold = 1:3
                    valIdx = test(c,fold);
                    trainIdx = valIdx == 0;
                    wKRR = (kernel(X(trainIdx,:),X(trainIdx,:),featuresFull)+lambda*sum(trainIdx)*eye(sum(trainIdx)))\Y(trainIdx);
                    valError(lambdaIdx,PIdx,fold) = mean((Y(valIdx)-real(kernel(X(valIdx,:),X(trainIdx,:),featuresFull)*wKRR)).^2);
                end
            end
        end
        % Pick best lambda
        valError = mean(valError,3);
        [~,minIdx] = min(valError,[],'all');
        [minLambdaIdx,minPIdx] = ind2sub(size(valError),minIdx);
        lambda = lambdaSet(minLambdaIdx);
        P = PSet(minPIdx);
        disp("Lambda: "+string(lambda));
        disp("P: "+string(P));
        % Select features
        features = @(X,m) FourierFeatures(X,m,M,P);
        featuresFull = @(X) FourierFeaturesFull(X,M,P); 
        % KRR final model
        tic;
        wKRR = (kernel(X,X,featuresFull)+lambda*N*eye(N))\Y;
        timeWallKRR = toc;
        trainErrorKRR = mean((Y-real(kernel(X,X,featuresFull)*wKRR)).^2);
        testErrorKRR = mean((YTest-real(kernel(XTest,X,featuresFull)*wKRR)).^2);

        for maxIte = maxIteSet
        ID = "val"+extractBefore(filename,".")+"M"+string(M)+"maxIte"+string(maxIte);
        disp(ID);
        % Determine maximum ranks
        [N, D] = size(readmatrix("./datasets/"+string(filename)));
        N = floor(0.8*N);
        D = D-1;
        maximumRankFF = 6;
        maximumRankQFF = ceil(maximumRankFF*M/log2(M)/2);
        maximumMRFF = maximumRankQFF*2*log2(M)*D;
        disp('Maximum rank FF: '+string(maximumRankFF));
        disp('Maximum rank QFF: '+string(maximumRankQFF));
        % Initialialize containers
        trainErrorFull = zeros(NTrials,maximumRankFF);
        testErrorFull = zeros(NTrials,maximumRankFF);
        timeWallFull = zeros(NTrials,maximumRankFF);
        trainErrorQuant = zeros(NTrials,maximumRankQFF);
        testErrorQuant = zeros(NTrials,maximumRankQFF);
        timeWallQuant = zeros(NTrials,maximumRankQFF);
        lossFull = zeros(NTrials,maximumRankFF,maxIte*D);
        lossQuant = zeros(NTrials,maximumRankQFF,maxIte*log2(M)*D);
        trainErrorRFF = zeros(NTrials,maximumMRFF);
        testErrorRFF = zeros(NTrials,maximumMRFF);
        timeWallRFF = zeros(NTrials,maximumMRFF);
        trainErrorComplexRFF = zeros(NTrials,maximumMRFF);
        testErrorComplexRFF = zeros(NTrials,maximumMRFF);
        timeWallComplexRFF = zeros(NTrials,maximumMRFF);

        for trial = 1:NTrials                
            % FF
            for R = 1:maximumRankFF
                rng(trial);
                tic;
                [WCPFull,lossFull(trial,R,:),~,~] = CPLSFullHighMemory(X,Y,featuresFull,M,R,lambda,maxIte);
                timeWallFull(trial,R) = toc;
                trainErrorFull(trial,R) = mean((Y-CPPredictFull(X,featuresFull,WCPFull)).^2);
                testErrorFull(trial,R) = mean((YTest-CPPredictFull(XTest,featuresFull,WCPFull)).^2);
            end
            % QFF
            for R = 1:maximumRankQFF
                rng(trial);
                tic;
                [WCP,lossQuant(trial,R,:),~,~] = CPLS(X,Y,features, M,R,lambda,maxIte);
                timeWallQuant(trial,R) = toc;
                trainErrorQuant(trial,R) = mean((Y-CPPredict(X,features,WCP)).^2);
                testErrorQuant(trial,R) = mean((YTest-real(CPPredict(XTest,features,WCP))).^2);
            end
            % RFF
            for MRFF = 1:maximumMRFF
                rng(trial);
                tic;
                [ZZ,ZY,W,B] = RFF(X,Y,MRFF,M/P/2);
                wRFF = (ZZ+N*lambda*eye(MRFF))\(ZY);
                timeWallRFF(trial,R) = toc;
                trainErrorRFF(trial,MRFF) = mean((Y-(RFFPredict(X,W,B)*wRFF)).^2);
                testErrorRFF(trial,MRFF) = mean((YTest-(RFFPredict(XTest,W,B)*wRFF)).^2);
            end
            % RFF
            for MRFF = 1:maximumMRFF
                rng(trial);
                tic;
                [ZZ,ZY,W,B] = ComplexRFF(X,Y,MRFF,M/P);
                wRFF = (ZZ+N*lambda*eye(MRFF))\(ZY);
                timeWallComplexRFF(trial,R) = toc;
                trainErrorComplexRFF(trial,MRFF) = mean((Y-real(ComplexRFFPredict(X,W,B)*wRFF)).^2);
                testErrorComplexRFF(trial,MRFF) = mean((YTest-real(ComplexRFFPredict(XTest,W,B)*wRFF)).^2);
            end

            save("./workspaces/"+ID+".mat",'N','D','M','P','lambda','maxIte','maximumRankFF','maximumRankQFF','maximumMRFF','NTrials','lossQuant','lossFull','trainErrorQuant','trainErrorFull','trainErrorRFF','trainErrorComplexRFF','trainErrorKRR','testErrorQuant','testErrorFull','testErrorRFF','testErrorComplexRFF','testErrorKRR');
        end
        errorbar((1:maximumRankFF)*D*M,mean(trainErrorFull),std(trainErrorFull));
        hold on
        errorbar(2*(1:maximumRankQFF)*D*log2(M),mean(trainErrorQuant),std(trainErrorQuant));
        hold on
        errorbar((1:maximumMRFF),mean(trainErrorRFF),std(trainErrorRFF));
        hold on
        errorbar((1:maximumMRFF),mean(trainErrorComplexRFF),std(trainErrorComplexRFF));
        hold on
        yline(trainErrorKRR);
        hold off
        set(gca,'XScale','log','YScale','log');
        xlabel('\# parameters','interpreter','latex');
        ylabel('train MSE','interpreter','latex');
        xlim([1,maximumMRFF]);
        legend('TFF','TQFF','RFF','ComplexRFF','KRR');
        axis square;
        exportgraphics(gca,"train"+string(ID)+".png");
        close all;
        
        errorbar((1:maximumRankFF)*D*M,mean(testErrorFull),std(testErrorFull));
        hold on
        errorbar(2*(1:maximumRankQFF)*D*log2(M),mean(testErrorQuant),std(testErrorQuant));
        hold on
        errorbar((1:maximumMRFF),mean(testErrorRFF),std(testErrorRFF));
        hold on
        errorbar((1:maximumMRFF),mean(testErrorComplexRFF),std(testErrorComplexRFF));
        hold on
        yline(testErrorKRR);
        hold off
        set(gca,'XScale','log','YScale','log');
        xlabel('\# parameters','interpreter','latex');
        ylabel('test MSE','interpreter','latex');
        xlim([1,maximumMRFF]);
        legend('TFF','TQFF','RFF','ComplexRFF','KRR');
        axis square;
        exportgraphics(gca,"test"+string(ID)+".png");
        close all;
        end
    end
end