clear
%%%%%%% STEP I, Load Training Data
X_Train_Raw = readtable('TrainData_4_2c.csv');
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

X_Test_Raw = readtable('TestData_4_2c.csv');
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


X_CV_Raw = readtable('CVData_4_2c.csv');
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

%%%%% LR
num_labels = 2; % Y=1, Y=2
lambda = 1;
[all_theta] = lrClassifier(X_train, Y_train, num_labels, lambda);
pred = predict(all_theta, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_train)) * 100);
%54.38

input_layer_size = size(X_train,2);
hidden_layer_size = 132;
num_labels = 2;
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter',50);
lambda = 1;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, Y_train, lambda);
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

pred = nnpredict(Theta1, Theta2, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_train)) * 100);
%55.28%

pred = nnpredict(Theta1, Theta2, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == Y_test)) * 100);

%51% 

pred = nnpredict(Theta1, Theta2, X_CV);
fprintf('\nCV Set Accuracy: %f\n', mean(double(pred == Y_CV)) * 100);
%52