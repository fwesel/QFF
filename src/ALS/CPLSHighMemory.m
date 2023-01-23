function [W, loss, error,timesteps] = CPLSHighMemory(X, y, features, M, R, lambda, numberSweeps)
    tic;
    [N, D] = size(X);
    log2M = floor(log2(M));
    W = cell(log2M,D);
    Mati = cell(log2M,D);
    Matd = 1;
    reg = 1;
    loss = zeros(numberSweeps*numel(W),1);
    error = zeros(numberSweeps*numel(W),1);
    timesteps = zeros(numberSweeps*numel(W),1);
    for d = D:-1:1
        for log2m = log2M:-1:1
            W{log2m,d} = randn(2,R);
            W{log2m,d} = W{log2m,d}./vecnorm(W{log2m,d},2,1);
            reg = reg.*(W{log2m,d}'*W{log2m,d});
            Mati{log2m,d} = features(X(:,d),log2m);
            Matd = (Mati{log2m,d}*W{log2m,d}).*Matd;
        end
    end
    f = waitbar(0,'Starting...');
    plotIdx = 0;
    for sweep = 1:numberSweeps
        for idx = 1:numel(W)
            plotIdx = plotIdx+1;
            [log2m,d] = ind2sub(size(W),idx);
            reg = reg./(W{log2m,d}'*W{log2m,d});
            Matd = Matd./(Mati{log2m,d}*W{log2m,d});
            C = dotkron(Mati{log2m,d},Matd);
            regularization = lambda*kron(reg,eye(2));
            if lambda == 0
                x = C\y;
            else
                x = (C'*C + N*regularization)\(C'*y);
            end
            loss(plotIdx) = norm(real(C*x)-y)^2+abs(x'*regularization*x);
            error(plotIdx) = mean((real(C*x)-y).^2);
            W{log2m,d} = reshape(x,2,R);
            scaling = vecnorm(W{log2m,d},2,1);
            W{log2m,d} = W{log2m,d}./scaling;
            reg = reg.*(W{log2m,d}'*W{log2m,d});
            Matd = Matd.*(Mati{log2m,d}*W{log2m,d});
            timesteps(plotIdx) = toc;
            waitbar(plotIdx/(numberSweeps*numel(W)),f,string(plotIdx/(numberSweeps*numel(W)))+"% complete, error: "+string(error(plotIdx)));
        end
    end
    W{log2m,d} = W{log2m,d}.*scaling;
    close(f);
end