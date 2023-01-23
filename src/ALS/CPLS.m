function [W, loss, error, time, lossVal] = CPLS(X, y, features, M, R, lambda, numberSweeps, varargin)



    tic;
    [N, D] = size(X);
    log2M = floor(log2(M));
    W = cell(log2M,D);
    loss = zeros(numberSweeps*numel(W),1);
    error = zeros(numberSweeps*numel(W),1);
    time = zeros(numberSweeps*numel(W),1);

    if ~isempty(varargin)
        XVal = varargin{1};
        YVal = varargin{2};
        lossVal = zeros(numberSweeps*numel(W),1);
    end


    Matd = 1;
    reg = 1;
    W = initQuantizedWeights(M,R,D);
    for d = D:-1:1
        for log2m = log2M:-1:1
            reg = reg.*(W{log2m,d}'*W{log2m,d});
            Mati = features(X(:,d),log2m);
            Matd = (Mati*W{log2m,d}).*Matd;
        end
    end
    
    plotIdx = 0;
    for sweep = 1:numberSweeps
        for idx = 1:numel(W)
            plotIdx = plotIdx+1;
            [log2m,d] = ind2sub(size(W),idx);
            Mati = features(X(:,d),log2m);
            reg = reg./(W{log2m,d}'*W{log2m,d});
            Matd = Matd./(Mati*W{log2m,d});
            C = dotkron(Mati,Matd);
            regularization = lambda*kron(reg,eye(2));
            if lambda == 0
                x = C\y;
            else
                x = (C'*C + N*regularization)\(C'*y);
            end
            contraction = C*x;
            loss(plotIdx) = mean((abs(contraction-y).^2))+abs(x'*regularization*x);
            error(plotIdx) = mean((real(C*x)-y).^2);
%             error(plotIdx) = mean(sign(real(contraction))~=y);
            W{log2m,d} = reshape(x,2,R);
            scaling = vecnorm(W{log2m,d},2,1);
            if ~isempty(varargin)
                lossVal(plotIdx) = mean((abs(CPPredict(XVal,W)-YVal)).^2) + abs(x'*regularization*x);
            end
            W{log2m,d} = W{log2m,d}./scaling;
            reg = reg.*(W{log2m,d}'*W{log2m,d});
            Matd = Matd.*(Mati*W{log2m,d});
            time(plotIdx) = toc;
        end
    end
    W{log2m,d} = W{log2m,d}.*scaling;
end