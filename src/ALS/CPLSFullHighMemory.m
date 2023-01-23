function [W, loss, error, timesteps] = CPLSFullHighMemory(X, y, featuresFull, M, R ,lambda, numberSweeps) 
    tic;
    [N, D] = size(X);
    W = cell(1,D);
    Mati = cell(1,D);
    Matd = 1;
    reg = 1;
    loss = zeros(numberSweeps*numel(W),1);
    timesteps = zeros(numberSweeps*numel(W),1);
    error = zeros(numberSweeps*numel(W),1);
    W = initFull(M,R,D);
    for d = D:-1:1
        reg = reg.*(W{d}'*W{d});
        Mati{d} = featuresFull(X(:,d));
        Matd = (Mati{d}*W{d}).*Matd;
    end
    idx = 0;
    for ite = 1:numberSweeps
        for d = 1:D
            idx = idx+1;
            reg = reg./(W{d}'*W{d});
            Matd = Matd./(Mati{d}*W{d});
            C = dotkron(Mati{d},Matd);
            regularization = lambda*kron(reg,eye(M));
            if lambda == 0
                x = C\y;
            else
                x = (C'*C + N*regularization)\(C'*y);
            end            
            loss(idx) = mean(abs(C*x-y).^2)+norm(x'*regularization*x)^2;
            error(idx) = mean((real(C*x)-y).^2);
            W{d} = reshape(x,M,R);
            scaling = vecnorm(W{d},2,1);
            W{d} = W{d}./scaling;
            reg = reg.*(W{d}'*W{d});
            Matd = Matd.*(Mati{d}*W{d});  
            timesteps(idx) = toc;
        end
    end
    W{d} = W{d}.*scaling;
end