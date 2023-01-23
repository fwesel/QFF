function score = CPPredictFull(X, featuresFull, W)
    [N,D] = size(X);
    score = ones(N,1);
    for d = 1:D
        score = score.*(featuresFull(X(:,d))*W{d});
    end
    score = real(sum(score,2));
end