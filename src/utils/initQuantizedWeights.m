function W = initQuantizedWeights(M,R,D)
    log2M = log2(M);
    W = cell(log2M,D);
    for factor = 1:numel(W)
        W{factor} = randn(2,R);
        W{factor} = W{factor}./vecnorm(W{factor},2,1);
    end
end