function [f,g] = logistic_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A column vector containing the parameter values to optimize.
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %

  m=size(X,2);
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));


  %
  % TODO:  Compute the objective function by looping over the dataset and summing
  %        up the objective values for each example.  Store the result in 'f'.
  %
  % TODO:  Compute the gradient of the objective by looping over the dataset and summing
  %        up the gradients (df/dtheta) for each example. Store the result in 'g'.
  %
%%% YOUR CODE HERE %%%
  n = size(X,1);
  h = zeros(m);
  for j = 1:m
      for k = 1 : n
        h(j) = h(j) + theta(k) * X(k, j);
      end
      h(j) = sigmoid(h(j));
      f = f + (1/2)*(h(j)-y(j))^2;
  end
  
  for i = 1:n
    g(i) = 0;
    for j = 1:m
      g(i) = g(i) +  X(i,j) * (h(j) - y(j));
      %theta(i)= theta(i) - g(i);
    end
  end