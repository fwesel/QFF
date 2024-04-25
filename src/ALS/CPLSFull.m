function [W, loss, error,time] = CPLSFull(X, y, featuresFull, M, R ,lambda, numberSweeps)
    % This function is part of the paper:
    % "Quantized Fourier and Polynomial Features for more Expressive Tensor Network Models"
    % by Frederiek Wesel and Kim Batselier, presented at the International Conference on Artificial Intelligence and Statistics, 2024.
    tic;    
    [N, D] = size(X);
    % Initialize projected features Matd and regularization term reg
    Matd = 1;
    reg = 1;
    W = initFull(M,R,D);
    for d = D:-1:1
        reg = reg.*(W{d}'*W{d});
        Mati = featuresFull(X(:,d));
        Matd = (Mati*W{d}).*Matd;
    end
    % Initialize containers
    loss = zeros(numberSweeps*numel(W),1);
    error = zeros(numberSweeps*numel(W),1);
    time = zeros(numberSweeps*numel(W),1);
    % Start ALS iterations
    plotIdx = 0;
    for ite = 1:numberSweeps
        for d = 1:D
            plotIdx = plotIdx+1;
            % Compute feature corresponding to d-th dimension
            Mati = featuresFull(X(:,d));
            % Prepare Matd and reg for update
            reg = reg./(W{d}'*W{d});
            Matd = Matd./(Mati*W{d});
            C = dotkron(Mati,Matd);
            regularization = lambda*kron(reg,eye(M));
            % Solve least-squares problem
            if lambda == 0
                x = C\y;
            else
                x = (C'*C + N*regularization)\(C'*y);
            end            
            contraction = C*x;
            loss(plotIdx) = mean((real(contraction)-y).^2)+abs(x'*regularization*x);
%             error(idx) = mean(((C*x)-y).^2);
            error(plotIdx) = mean(sign(real(contraction))~=y);
            % Reshape d-th core and update Matd and reg
            W{d} = reshape(x,M,R);
            scaling = vecnorm(W{d},2,1);
            W{d} = W{d}./scaling;
            reg = reg.*(W{d}'*W{d});
            Matd = Matd.*(Mati*W{d});  
            time(plotIdx) = toc;
        end
    end
    W{d} = W{d}.*scaling;
end