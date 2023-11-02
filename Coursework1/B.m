clear all;                              % Clear and close all previous 
close all;                              % variables and figures

load cw1a.mat;                          % Load in the data for training
xs = linspace(-4, 4, 201)';              % 61 test inputs

meanfunc = [];           % empty: don't use a mean function
covfunc = @covSEiso;     % Squared Exponental covariance function
likfunc = @likGauss;     % Gaussian likelihood   


num_test = 3;
evidence = zeros(num_test);
hyp_opts = zeros(num_test , 3);
titles = ["First Local Optimum", "Second Local Optimum","Third Local Optimum"];
params = [[-1, 0, 0]; [15, 0, 0]; [0, 0, 1]]; % [1, 0, 0]; [0, 1, 0];
figure;
hold on;
for i = 1:num_test
    hyp_init = struct('mean', [], 'cov', [params(i, 1), params(i, 2)], 'lik', params(i, 3));
    hyp_opt = minimize(hyp_init, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    evidence(i + 1) = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    hyp_opts(i + 1, :) = [hyp_opt.cov(1), hyp_opt.cov(2), hyp_opt.lik];
    subplot(num_test, 1, i);
    [mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2), 1)];
    fill([xs; flipdim(xs,1)], f, [7 7 7]/8);
    hold on;
    title(titles(:, i)); 
    xlabel('X'); ylabel('Y');
    plot(xs, mu); plot(x, y, '+');
end
hold off;
disp(hyp_opts);
disp(evidence);