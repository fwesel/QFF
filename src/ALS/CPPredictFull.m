function score = CPPredictFull(X, featuresFull, W)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    [N,D] = size(X);
    score = ones(N,1);
    for d = 1:D
        score = score.*(featuresFull(X(:,d))*W{d});
    end
    % Ensure real-valued prediction
    score = real(sum(score,2));
end