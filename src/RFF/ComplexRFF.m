function [ZZ,ZY,W,B] = ComplexRFF(X,Y,M,MMax)
    batchSize = 100;
    [N,D] = size(X);
    W = 2*pi*(rand(D,M)-0.5)*MMax;
    B = 2*pi*rand(1,M);
    ZZ = zeros(M,M);
    ZY = zeros(M,1);
    for n = 1:batchSize:N
        idx = min(n+batchSize-1,N);
        temp = sqrt(1/M)*exp(1i*X(n:idx,:)*W+B);
        ZZ = ZZ+temp'*temp;
        ZY = ZY+temp'*Y(n:idx);
    end
end