function Mati = purePowerFeatures(X,m)
    Mati = [ones(size(X)), X.^(2.^(m-1))];
end