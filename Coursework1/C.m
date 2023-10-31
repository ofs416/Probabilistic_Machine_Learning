clear all;                              % Clear and close all previous 
close all;                              % variables and figures

load cw1a.mat;                          % Load in the data for training
xs = linspace(-4, 4, 31)';              % 61 test inputs

meanfunc = []; hyp.mean = [];           % empty: don't use a mean function
covfunc = @covPeriodic; hyp.cov = [0 0 0];  % Periodic covariance function
likfunc = @likGauss; hyp.lik = 0;       % Gaussian likelihood   

disp(' ');
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, ...
                likfunc, x, y)

disp(' ');
nlml = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)


[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);



f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2), 1)];
     fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
     hold on; plot(xs, mu); plot(x, y, '+');