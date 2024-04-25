function Mati = FourierFeatures(X,m,M,P)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    Mati = [exp(-1i*pi*X*M/(log2(M)*P)), exp(1i*pi*(-X*M/log2(M)+2*X.*(2^(m-1)))/P)];
    % Quantized Fourier Features, with Q=2 as described in the paper.
end