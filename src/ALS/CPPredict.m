function score = CPPredict(X, features, W)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    [N,D] = size(X);
    log2M = size(W,1);
    score = ones(N,1);
    for d = 1:D
        for log2m = 1:log2M
            Mati = features(X(:,d),log2m);
            score = score.*(Mati*W{log2m,d});
        end
    end
    % Ensure real-valued prediction
    score = real(sum(score,2));
end