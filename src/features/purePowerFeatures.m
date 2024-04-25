function Mati = purePowerFeatures(X,m)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    Mati = [ones(size(X)), X.^(2.^(m-1))];
    % Quantized Pure-Power Polynomial Features, with Q=2 as described in the paper.
end