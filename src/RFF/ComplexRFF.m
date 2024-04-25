function [ZZ,ZY,W,B] = ComplexRFF(X,Y,M,MMax)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    % This function implements complex-valued random features for large-scale kernel machines, as described in the paper:
    % "Random features for large-scale kernel machines"
    % by Ali Rahimi and Benjamin Recht, published in Advances in Neural Information Processing Systems, 2007.
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