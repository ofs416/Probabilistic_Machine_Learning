clear all;                              % Clear and close all previous 
close all;                              % variables and figures

load cw1a.mat;                          % Load in the data for training
xs = linspace(-4, 4, 101)';              % 61 test inputs

meanfunc = []; hyp.mean = [];           % empty: don't use a mean function
covfunc = @covPeriodic; hyp.cov = [0 0 0];  % Periodic covariance function
likfunc = @likGauss; hyp.lik = 0;       % Gaussian likelihood   

hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, ...
                likfunc, x, y)
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);


figure()
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2), 1)];
     fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
     title('Predictive Mean and 95% Confindence Bands'); 
     xlabel('Input, X'); ylabel('Output, Y');
     hold on; plot(xs, mu); 
     g = plot(x, y, '+', 'DisplayName', 'Training Data');
     legend(g);



meanfunc = [];          % empty: don't use a mean function
covfunc = {@covSum, {@covSEard, @covPeriodic}};  % Periodic covariance function
likfunc = @likGauss;      % Gaussian likelihood   
hyp = struct('mean', [], 'cov', [0 0 0 0 0], 'lik', -1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, ...
                likfunc, x, y)
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

figure()
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2), 1)];
     fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
     title('Predictive Mean and 95% Confindence Bands'); 
     xlabel('Input, X'); ylabel('Output, Y');
     hold on; plot(xs, mu);
     g = plot(x, y, '+', 'DisplayName', 'Training Data');
     legend(g);