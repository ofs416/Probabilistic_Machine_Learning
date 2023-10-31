clear all;                              % Clear and close all previous 
close all;                              % variables and figures


covfunc = {@covProd, {@covPeriodic, @covSEiso}};
hyp.cov = [-0.5 0 0 2 0];               % Periodic covariance function

x = linspace(-5,5,200)';
K = feval(covfunc{:}, hyp.cov, x); C = [1e-6*eye(200)]*200;
y = chol(K + C)'*gpml_randn(0.2, 200, 2);

hold on;;plot(x, y);