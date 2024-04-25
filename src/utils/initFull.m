% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function W = initFull(M,R,D)
    % This function initializes the cores of the tensor network kernel
    % machine (TKM) based on the initialization of the quantized tensor
    % network kernel machine (QTKM) in order to ensure same initial guess.
    WQ = initQuantizedWeights(M,R,D);
    W = cell(D,1);
    for d = 1:D
       W{d} = ones(M,R);
        for r = 1:R
            temp = 1;
            for log2m = 1:log2(M)
               temp = kron(temp, WQ{log2m,d}(:,r));
            end
            W{d}(:,r) = temp;
        end
    end
end