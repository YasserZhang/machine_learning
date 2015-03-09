function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
n = size(Theta2,1);
a1 = [ones(m, 1) X];
z2 = Theta1*transpose(a1);
a2 = sigmoid(z2);
a2 = [ones(1,size(a2, 2)); a2];
z3 = Theta2*a2;
a3 = sigmoid(z3); % it is a matrix
c = [1:n];
y1 = zeros(m,n);
for i = 1:m
  y1(i,:) = (y(i) == c);
end
y1 = transpose(y1);
J = -sum(sum((log(a3).*y1 + log(1-a3).*(1-y1))))/m;

theta_reg1 = Theta1;
theta_reg1(:,1) = 0;
theta_reg2 = Theta2;
theta_reg2(:,1) = 0;

J = J + (sum(sum(theta_reg1 .^ 2)) + sum(sum(theta_reg2 .^ 2)))*lambda/(2*m);

for i = 1:m
  a1 = transpose([1 X(i,:)]); % 401 x 1;
  z2 = Theta1 * a1; % 25 x 1;
  a2 = sigmoid(z2); % 25 x 1;
  a2 = [1; a2];     % 26 x 1;
  z3 = Theta2 * a2; % 10 x 1;
  a3 = sigmoid(z3); % 10 x 1;
  
  d3 = a3 - y1(:, i);  % 10 x 1;
  
  del = transpose(Theta2) * d3; % 26 x 1;
  d2 = del(2:end).*sigmoidGradient(z2); % 25 x 1;
  Theta2_grad = Theta2_grad + d3 * transpose(a2); % 10 x 26
  Theta1_grad = Theta1_grad + d2 * transpose(a1); % 25 x 401
end
t2 = Theta2;
t2(:,1) = 0;
t1 = Theta1;
t1(:,1) = 0;
Theta2_grad = (Theta2_grad + lambda*t2)/m;
Theta1_grad = (Theta1_grad + lambda*t1)/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
