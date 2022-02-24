function [p] = predict(all_theta, X)
%%PREDICT Predict the label for a trained one-vs-all classifier.
%  p = PREDICT(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class.


% initilize values
m = size(X, 1);

num_labels = size(all_theta, 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

n = size(X,2);
new_all_theta = all_theta(:,1:n);
prediction_matrix1 = X * new_all_theta';
prediction_matrix = 1 ./ (1 + exp(-prediction_matrix1));
[v,p] = max(prediction_matrix,[],2);




end

