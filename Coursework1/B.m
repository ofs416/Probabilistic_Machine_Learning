clear all;                              % Clear and close all previous 
close all;                              % variables and figures

load cw1a.mat;                          % Load in the data for training
xs = linspace(-4, 4, 201)';              % 61 test inputs

meanfunc = [];           % empty: don't use a mean function
covfunc = @covSEiso;     % Squared Exponental covariance function
likfunc = @likGauss;     % Gaussian likelihood   


num_test = 2;
evidence = zeros(num_test);
hyp_opts = zeros(num_test , 3);
titles = ["First Local Optimum", "Second Local Optimum","Third Local Optimum"];
params = [[-1, 0, 0]; [2, -0.36, -0.4]]; % [1, 0, 0]; [0, 1, 0];
figure()
hold on;
for i = 1:num_test
    hyp_init = struct('mean', [], 'cov', [params(i, 1), params(i, 2)], 'lik', params(i, 3));
    hyp_opt = minimize(hyp_init, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    evidence(i) = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    hyp_opts(i, :) = [hyp_opt.cov(1), hyp_opt.cov(2), hyp_opt.lik];
    subplot(num_test, 1, i);
    [mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2), 1)];
    fill([xs; flipdim(xs,1)], f, [7 7 7]/8);
    hold on;
    title(titles(:, i)); 
    xlabel('Input, X'); ylabel('Output, Y');
    plot(xs, mu);
    g = plot(x, y, '+', 'DisplayName', 'Training Data');
    legend(g);
end
hold off;
disp(hyp_opts);
disp(evidence);

N = 100; a = 3.5;
Xs = linspace(-1.5*a,0.75*a,N); Ys = linspace(-2*a,1*a,N);
Z = zeros(N,N); 

for i = 1:N
    for j = 1:N
        hyp = struct('mean', [], 'cov', [Xs(i) 0], 'lik', Ys(j));
        [nlml, dnlml] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
        Z(j, i) = log(nlml);
    end
end

figure()
contourf(Xs, Ys, Z, 10)
colormap("parula")
cb = colorbar;
cb.Label.String = 'Log of the Negative Log Marginal Likelihood';
title('Contour Plot showing Optimal NLML Hyperparameters'); 
xlabel('Log(Length Scale)'); ylabel('Log(S.D. of Noise)');




