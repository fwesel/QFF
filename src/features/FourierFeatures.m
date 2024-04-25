% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function Mati = FourierFeatures(X,m,M,P)
    % Quantized Fourier Features, with Q=2 as described in the paper based
    % on the Fourier Features as described in:
    % "Learning multidimensional Fourier series with tensor trains"
    % by Sander Wahls, Visa Koivunen, H Vincent Poor, and Michel Verhaegen, presented at the 2014 IEEE Global Conference on Signal and Information Processing (GlobalSIP).
    Mati = [exp(-1i*pi*X*M/(log2(M)*P)), exp(1i*pi*(-X*M/log2(M)+2*X.*(2^(m-1)))/P)];
end