function [J,grad] = lrCostFunction(theta,X,y,lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


%Initialize some useful values
m = length(y);
grad = zeros(size(theta));

%I. Cost Function J: notice that i starts from 1 not 0
% 1) Find h which is the hypothesis value
theta_no0 = theta(2:end);
%sigmoid function g(z) = 1 / (1 +e^(-z))
% z = theta0*X0 + theta1*X1 + ..........
%theta is a column vector eg. [1;2;3;4;5....]
z_mat = X .* (theta)';
z = sum(z_mat,2); %sum(A,2) is a column vector containing the sum of each row.
h = 1./(1 + exp(-z));

%2) Find unregularized Cost function J
J_part1_1 = -(y .* log(h));
J_part1_2 = (1 - y) .* log(1 - h);
J_part1 = (1/m) * sum(J_part1_1 - J_part1_2);

%3) Find regularized Cost function J: add (lambda/2m) * sum(theta^2)
% sum starts from 1 == exclude theta0 since X0 = 1 
J_part2 = (lambda / (2*m)) * sum(theta_no0 .* theta_no0);
J = J_part1 + J_part2;



%4) Find unregularized Gradient descent 
% grad_theta_j = (1/m) * sum (h(xi) - y(i)) * x(i,j)
% sum starts from i = 1
grad_unreg = (1/m) * ((X)' * (h - y));

%5) Find regularized Grad descent (do not regularize theta0)
grad_reg = grad_unreg + (lambda / m) * theta;
grad = [grad_unreg(1); grad_reg(2:end)];
end

