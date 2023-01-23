addpath('./src/ALS')
addpath('./src/features')
addpath('./src/utils')
close all
warning('off','all');
%% Yacht Dataset
RSet = [10,25];
P = 1;
lambda = 0;
maxIte = 5000;
loss = cell(numel(RSet),1);
warning('off','all');
rng('default');
X = load('datasetsStorage/audio_data.mat'); 
YTest = X.ytest; 
XTest= X.xtest;
Y = X.ytrain;  
X = X.xtrain;

YMean = mean(Y);    YStd = std(Y);
XMin = min(X);  XMax = max(X);
Y = (Y-YMean)./YStd;
X = (X-XMin)./(XMax-XMin)-0.5; 
XTest = (XTest-XMin)./(XMax-XMin)-0.5;
YTest = (YTest-YMean)./YStd;
samplingFrequency = 1/min(diff(X));
M = samplingFrequency/2;
M = 2^floor(log2(M));
M = M/2;

% Fourier features
featuresFull = @(X,m) FourierFeaturesFull(X,M,P);
features = @(X,m) FourierFeatures(X,m,M,P);

% Compute Fourier coefficients
w = featuresFull(X,M)\Y;
SMAEFull = mean(abs((Y-real(featuresFull(X,M)*w))));
SMAEFullTest = mean(abs(YTest-real(featuresFull(XTest,M)*w)));
disp('Full weights computed');

ite = 0;
for R = RSet
disp('R: '+string(R));
ite = ite+1;
rng('default');

% Compute CPD-rank-R Fourier coefficients
tic;
[WCP,loss{ite},~,~] = CPLS(X, Y, features, M, R ,lambda, maxIte);
SMAECP(ite) = mean(abs(Y-real(CPPredict(X,features,WCP))));
SMAETestCP(ite) = mean(abs(YTest-real(CPPredict(XTest,features,WCP))));
wReconstructed = ones(R,1);

% Reconstruct weights
for m = 1:log2(M)
    wReconstructed = dotkron(wReconstructed,transpose(WCP{m}));
end
wReconstructed = transpose(sum(wReconstructed));

% Relative error and compression ratio
relativeError = norm(wReconstructed-w,'fro')/norm(w,'fro');
compressionRatio = 2*log2(M)*R/numel(w);

% Plot
close all;
b1 = bar(0:M/2-1,abs(w(M/2+1:end)),'black');
b1.FaceAlpha = 0.7;
hold on;
b2 = bar(0:M/2-1,abs(wReconstructed(M/2+1:end)));
b2.FaceAlpha = 0.7;
hold on;
hold off;
pbaspect([1 1.61803398875 1]);
set(gca,'XScale','linear','YScale','linear','LineWidth',1.5);
xlabel('$m$','interpreter','latex','FontSize',20);
ylabel('$\vert$\boldmath{$w$}{}\unboldmath$\vert$','interpreter','latex','FontSize',20);
xlim([0 M/2-1])
ylim([0 0.06])
title('$R=50$, $P=1300$','interpreter','latex','FontSize',20);
if ite == 1
    legend('Non-Quantized','2-Quantized','interpreter','latex','FontSize',15);
end

%Save
exportgraphics(gca,"FourierCoefficientsR"+string(R)+"ite"+string(maxIte)+".pdf");
savefig("FourierCoefficientsR"+string(R)+"ite"+string(maxIte)+".fig");
save("./workspaces/"+"FourierCoefficientsR"+string(R)+"ite"+string(maxIte)+".mat",'');
end