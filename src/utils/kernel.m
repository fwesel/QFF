function K = kernel(X,Z,features)
    [~,D] = size(X);
    K = 1;
    for d = 1:D
        K = K.*(features(X(:,d))*features(Z(:,d))');
    end
end