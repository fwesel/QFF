% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function W = initQuantizedWeights(M,R,D)
    % This function initializes the cores of the quantized tensor
    % network kernel machine (QTKM).
    log2M = log2(M);
    W = cell(log2M,D);
    for factor = 1:numel(W)
        W{factor} = randn(2,R);
        W{factor} = W{factor}./vecnorm(W{factor},2,1);
    end
end