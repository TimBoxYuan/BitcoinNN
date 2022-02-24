function [all_theta] = lrClassifier(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i

%1. Initilize some variables
m = size(X,1);
n = size(X,2);
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix, X0 = [1;1;1;1.....]
X = [ones(m, 1) X];

% Start Training
i = 1;
for c = (1:num_labels)
    initial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    [theta] = ...
        fmincg(@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
        initial_theta, options);
    all_theta(i,:) = theta;
    i = i + 1;
        
    
    
end

