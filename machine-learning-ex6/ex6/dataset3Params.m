function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


C_check = [0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];
sigma_check = [0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];

error_score = zeros(size(C_check),size(sigma_check));

for i=1:length(C_check),
  for j=1:length(sigma_check),
    model= svmTrain(X, y, C_check(i), @(x1, x2) gaussianKernel(x1, x2, sigma_check(j))); 
    predictions = svmPredict(model,Xval);
    error_score(i,j) = mean(double(predictions ~= yval));
  endfor
endfor

[v1 idx1] =  min(min(error_score,[],2));
C = C_check(idx1);
[v2 idx2] =  min(min(error_score,[],1));
sigma = sigma_check(idx2);


% =========================================================================

end
