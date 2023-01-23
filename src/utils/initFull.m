function W = initFull(M,R,D)
    WQ = initQuantizedWeights(M,R,D);
    W = cell(D,1);
    for d = 1:D
       W{d} = ones(M,R);
        for r = 1:R
            temp = 1;
            for log2m = 1:log2(M)
               temp = kron(temp, WQ{log2m,d}(:,r));
            end
            W{d}(:,r) = temp;
        end
    end
end