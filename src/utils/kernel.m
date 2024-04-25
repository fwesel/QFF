% This function is part of the paper:
% "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
% by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
function K = kernel(X,Z,features)
    % This function explicitly evaluates the kernel as inner product
    % between features.
    [~,D] = size(X);
    K = 1;
    for d = 1:D
        K = K.*(features(X(:,d))*features(Z(:,d))');
    end
end