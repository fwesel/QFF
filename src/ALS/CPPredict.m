function score = CPPredict(X, features, W)
    [N,D] = size(X);
    log2M = size(W,1);
    score = ones(N,1);
    M = 2^log2M;
    for d = 1:D
        for log2m = 1:log2M
            Mati = features(X(:,d),log2m);
            score = score.*(Mati*W{log2m,d});
        end
    end
    score = real(sum(score,2));
end