
% Neural Network X === 17 features
clear
%%%%%%% STEP I, Load Training Data
X_Train_Raw = readtable('TrainData.csv');

%Switch Data from table to Array for X_train
num_row = size(X_Train_Raw,1);
num_col = size(X_Train_Raw,2);
X_train = zeros(num_row, num_col - 2);%!!!!
j = 1;
for i = 2:(num_col-1)
    a = X_Train_Raw(:,i);
    A = table2array(a);
    X_train(:,j) = A;
    j = j + 1;
end

%Switch Data from table to Array for Y
a = X_Train_Raw(:,end);
Y_train = table2array(a); 

%%%%% STEP II: Load Testing Data

X_Test_Raw = readtable('TestData.csv');
%Switch Data from table to Array for X
num_row = size(X_Test_Raw,1);
num_col = size(X_Test_Raw,2);
X_test = zeros(num_row, num_col - 2);
j = 1;
for i = 2:(num_col-1)
    a = X_Test_Raw(:,i);
    A = table2array(a);
    X_test(:,j) = A;
    j = j + 1;
end
a = X_Test_Raw(:,end);
Y_test = table2array(a); 

%%%%%%STEP III: Load CV Data
X_CV_Raw = readtable('CVData.csv');
%Switch Data from table to Array for X
num_row = size(X_CV_Raw,1);
num_col = size(X_CV_Raw,2);
X_CV = zeros(num_row, num_col - 2);
j = 1;
for i = 2:(num_col-1)
    a = X_CV_Raw(:,i);
    A = table2array(a);
    X_CV(:,j) = A;
    j = j + 1;
end
a = X_CV_Raw(:,end);
Y_CV = table2array(a); 

X_Raw = readtable('RandomData.csv');

%Switch Data from table to Array for X_train
num_row = size(X_Raw,1);
num_col = size(X_Raw,2);
X_random = zeros(num_row, num_col - 2);%!!!!
j = 1;
for i = 2:(num_col-1)
    a = X_Raw(:,i);
    A = table2array(a);
    X_random(:,j) = A;
    j = j + 1;
end

%Switch Data from table to Array for Y
a = X_Raw(:,end);
Y_random = table2array(a);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FINISHED LOADING DATA%%%%%%%%%%%%%%%%%%%%%%

%I. Pick a Model
% 3 total layers: 1 input, 1 hidden, 1 output
% 17 inputs layer 1; 34 inputs layer 2; 3 outputs layer 3

% STEP I: Cost Function for Neural Network
% Create a function named nnCostFunction
%function [J grad] = nnCostFunction(nn_params,
                                   %input_layer_size,
                                   %hidden_layer_size,
                                   %num_labels,
                                   %X, y, lambda)

                                   
%STEP II: initialize Thetas (can't use [0,0,0])

%%%%Learning Parameters
input_layer_size = 17;
hidden_layer_size = 34;
num_labels = 3;

% Start a function called randInitializeWeights
%function W = randInitializeWeights(L_in, L_out)
%Try if your function works
initial_Theta1 = randInitializeWeights(17,34);
initial_Theta2 = randInitializeWeights(34,3);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];




options = optimset('MaxIter',50);
lambda = 0.1;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, Y_train, lambda);
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

pred = nnpredict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_train)) * 100);
% .  50.81%


pred = nnpredict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == Y_test)) * 100);
%49%

pred = nnpredict(Theta1, Theta2, X_CV);
fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == Y_CV)) * 100);
%52% 


%%%% YES We have a bias problem. Add polynomials 