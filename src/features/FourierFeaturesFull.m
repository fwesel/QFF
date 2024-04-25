function Mati = FourierFeaturesFull(X,M,P)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    Mati = exp(1j*2*pi*X.*(-M/2:(M/2-1))/P);
    % Fourier Features as described in:
    % "Learning multidimensional Fourier series with tensor trains"
    % by Sander Wahls, Visa Koivunen, H Vincent Poor, and Michel Verhaegen, presented at the 2014 IEEE Global Conference on Signal and Information Processing (GlobalSIP).
end