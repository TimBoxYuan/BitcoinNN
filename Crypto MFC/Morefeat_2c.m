clear
X_Train_Raw = readtable('FinalDataCollection_5_2c_more_feat.csv');

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

num_labels = 2; % Y=1, Y=2,3
lambda = 0.001;
[all_theta] = lrClassifier(X_train, Y_train, num_labels, lambda);
pred_log = predict(all_theta, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred_log == Y_train)) * 100);
%47.745%


input_layer_size = size(X_train,2);
hidden_layer_size = 105 ;
num_labels = 2;
initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size,num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

options = optimset('MaxIter',50);
lambda = 0.001;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, Y_train, lambda);
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

pred = nnpredict(Theta1, Theta2, X_train);
fprintf('\nLR Training Set Accuracy: %f\n', mean(double(pred_log == Y_train)) * 100);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_train)) * 100);