function [J, grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
    num_labels, X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification

%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Initialize variables
m = size(X,1);

%%%% Part 1 Find h (same as the logical regression h, 2 layers)
X = [ones(m,1), X]; %add 1 column to X
layer2_1 = X * Theta1';
layer2_2 = 1 ./ (1 + exp(-layer2_1));
k = size(layer2_2,1);
layer2 = [ones(k,1), layer2_2]; %add 1 column to Layer2 outputs
layer3_1 = layer2 * Theta2';
layer3 = 1 ./ (1 + exp(-layer3_1));
h = layer3;
n = size(layer3,2);


Y = zeros(k,n);
for i = 1:m
    V = Y(i,:);
    V(y(i)) = 1;
    Y(i,:) = V;
end

%Find J unregularized
J_part1 = - Y .* log(h);
J_part2 = (1 - Y) .* log(1 - h);
J_1 = (1/m) * sum(sum(J_part1 - J_part2,2));
%Find J regularized 
real_Theta1 = Theta1( : , 2 : end); 
real_Theta2 = Theta2(: , 2:end);
theta1_square = real_Theta1 .* real_Theta1;
theta2_square = real_Theta2 .* real_Theta2;
J_reg1 = sum(sum(theta1_square,2));
J_reg2 = sum(sum(theta2_square,2));
J_reg = (lambda /(2*m)) * (J_reg1 + J_reg2);
J = J_1 + J_reg;

%Gradient


delta3 = h - Y;
size_l21 = size(layer2_1,1);
z2 = [ones(size_l21,1), layer2_1];
delta2_1 = (delta3 * Theta2) .* sigmoidGradient(z2);
delta2 = delta2_1(: , 2:end);
Delta_1 = delta2' * X;
Delta_2 = delta3' * layer2;
Theta1_grad_Unreg =  (1/m).* Delta_1;
Theta2_grad_Unreg = (1/m) .* Delta_2;
Theta1_grad_Reg_Noj0 = Theta1_grad_Unreg + (lambda/m).*Theta1;
Theta1_grad_Reg_Noj0(:,1) = Theta1_grad_Unreg(:,1);
Theta1_grad = Theta1_grad_Reg_Noj0;
Theta2_grad_Reg_Noj0 = Theta2_grad_Unreg + (lambda/m).*Theta2;
Theta2_grad_Reg_Noj0(:,1) = Theta2_grad_Unreg(:,1);
Theta2_grad = Theta2_grad_Reg_Noj0;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

