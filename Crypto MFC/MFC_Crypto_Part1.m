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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FINISHED LOADING DATA%%%%%%%%%%%%%%%%%%%%%%


%%% Part 2 Cost Function

%X_train has 2100 examples, each row is an example. 
%There are 17 features in each example

% START a Function called: lrCostFunction
% function [J, grad] = lrCostFunction(theta, X, y, lamda)
% Compute Grad and J
% TEST the correctness of lrCostFunction
theta_t = [-2;-1;1;2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = [1;0;1;0;1];
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
fprintf('Cost: %f |\n',J); %Expected cost: 2.534819
fprintf('Gradients:\n'); 
fprintf('%f\n',grad); %0.146561,-0.548558,0.724722,1.398003
%%%%%%%%%% COST FUNCTION COMPLETE%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%% PART 3 Training Classifiers for each class [1,2,3]
% Start a Function lrClassifier
%function [all_theta] = lrClassifier(X, y, num_labels, lambda)
%TEST see if it runs
num_labels = 3; % Y=1, Y=2, Y=3
lambda = 0.1;
[all_theta] = lrClassifier(X_train, Y_train, num_labels, lambda);
%Mine works

%%%%% Part 4 See the prediction accuracy of our training results
% Start a function called predict
%function p = predic(all_theta, X)
pred = predict(all_theta, X_train);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == Y_train)) * 100);
% .  48.2% . Not Great

pred_test = predict(all_theta, X_test);
fprintf('\ntest Set Accuracy: %f\n', mean(double(pred_test == Y_test)) * 100);
% 49.76%

pred_cv = predict(all_theta, X_CV);
fprintf('\nCV Set Accuracy: %f\n', mean(double(pred_cv == Y_CV)) * 100);
% 53.43% this is cool


% MAY NEED TO ADD Polynomials X^2
%Remeber to record your results: we going back to re-clean data.
%% Test the entire data set for fun
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

num_labels = 3;
lambda = 0.05;
[all_theta] = lrClassifier(X_random, Y_random, num_labels, lambda);

pred = predict(all_theta, X_random);
fprintf('\n Total Set Accuracy: %f\n', mean(double(pred == Y_random)) * 100);

%51.157 % 


