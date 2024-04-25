% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function Mati = purePowerFeaturesFull(X,M)
    % Pure-Power Polynomial Features as described in:
    % "Parallelized tensor train learning of polynomial classifiers"
    % by Zhongming Chen, Kim Batselier, Johan AK Suykens, and Ngai Wong, published in the IEEE Transactions on Neural Networks and Learning Systems, 2017.
    Mati = X.^(0:M-1);
end
