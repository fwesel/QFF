function Z = ComplexRFFPredict(X,W,B)
    M = size(W,2);
    Z = sqrt(1/M)*exp(1i*X*W+B);
end