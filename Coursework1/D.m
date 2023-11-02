clear all;                              % Clear and close all previous 
close all;                              % variables and figures


covfunc = {@covProd, {@covPeriodic, @covSEiso}};
hyp.cov = [-0.5 0 0 2 0];             % Periodic covariance function

x = linspace(-5,5,400)';
K = feval(covfunc{:}, hyp.cov, x);
y = chol(K + 1e-6 * eye(400))' * randn(400, 3);

hold on; 
title(['GP generated Sample Functions']); 
xlabel('X'); ylabel('Y');
plot(x, y);