function Mati = FourierFeaturesFull(X,M,P)
    Mati = exp(1j*2*pi*X.*(-M/2:(M/2-1))/P);
end

