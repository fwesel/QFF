function Z = RFFPredict(X,W,B)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    % This function implements predictions for real-valued random features for large-scale kernel machines, as described in the paper:
    % "Random features for large-scale kernel machines"
    % by Ali Rahimi and Benjamin Recht, published in Advances in Neural Information Processing Systems, 2007.
    M = size(W,2);
    Z = sqrt(2/M)*cos(X*W+B);
end