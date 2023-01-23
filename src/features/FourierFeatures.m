function Mati = FourierFeatures(X,m,M,P)
    Mati = [exp(-1i*pi*X*M/(log2(M)*P)), exp(1i*pi*(-X*M/log2(M)+2*X.*(2^(m-1)))/P)];
end