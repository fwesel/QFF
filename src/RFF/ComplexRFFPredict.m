% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function Z = ComplexRFFPredict(X,W,B)
    % This function implements predictions for complex-valued random features for large-scale kernel machines, as described in the paper:
    % "Random features for large-scale kernel machines"
    % by Ali Rahimi and Benjamin Recht, published in Advances in Neural Information Processing Systems, 2007.
    M = size(W,2);
    Z = sqrt(1/M)*exp(1i*X*W+B);
end