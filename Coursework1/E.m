clear all;                              % Clear and close all previous 
close all;                              % variables and figures

load cw1e.mat;   

figure(1)
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11));

Bounds = 8;
Intervals = 0.1;
Steps = (2*Bounds) / Intervals + 1;

xs = [];
for i = -Bounds:Intervals:Bounds
    disp(i)
    for j = -Bounds:Intervals:Bounds
        xs(end+1, :) = [i ; j];
    end
end

xs1 = reshape(xs(:,1),Steps,Steps); xs2 = reshape(xs(:,2),Steps,Steps);

% Model 1 -----------------------------------------------------------------

meanfunc_1 = []; hyp_1.mean = [];        % empty: don't use a mean function
covfunc_1 = @covSEard; hyp_1.cov = [0 0 0];           
likfunc_1 = @likGauss; hyp_1.lik = 0;       % Gaussian likelihood   


hyp2_1 = minimize(hyp_1, @gp, -100, @infGaussLik, meanfunc_1, covfunc_1, ...
                  likfunc_1, x, y)
nlml_1 = gp(hyp2_1, @infGaussLik, meanfunc_1, covfunc_1, likfunc_1, x, y)


[mu_1, s2_1] = gp(hyp2_1, @infGaussLik, meanfunc_1, covfunc_1, likfunc_1, ...
                 x, y, xs);


figure(2)
mesh(xs1, xs2, reshape(mu_1+2*sqrt(s2_1), Steps, Steps),'FaceAlpha','0.05');
hold on;
mesh(xs1, xs2, reshape(mu_1-2*sqrt(s2_1), Steps, Steps),'FaceAlpha','0.05');
hold on;
surf(xs1, xs2, reshape(mu_1,Steps,Steps),'FaceAlpha','0.5');


% Model 2 -----------------------------------------------------------------

meanfunc_2 = []; hyp_2.mean = [];        % empty: don't use a mean function
covfunc_2 =  {@covSum, {@covSEard, @covSEard}}; hyp_2.cov = 0.1*randn(6,1); 
likfunc_2 = @likGauss; hyp_2.lik = 0;       % Gaussian likelihood   


hyp2_2 = minimize(hyp_2, @gp, -100, @infGaussLik, meanfunc_2, covfunc_2, ...
                  likfunc_2, x, y)
nlml_2 = gp(hyp2_2, @infGaussLik, meanfunc_2, covfunc_2, likfunc_2, x, y)


[mu_2, s2_2] = gp(hyp2_2, @infGaussLik, meanfunc_2, covfunc_2, likfunc_2, ...
                  x, y, xs);


figure(3)
mesh(xs1, xs2, reshape(mu_2+2*sqrt(s2_2), Steps, Steps),'FaceAlpha','0.05');
hold on;
mesh(xs1, xs2, reshape(mu_2-2*sqrt(s2_2), Steps, Steps),'FaceAlpha','0.05');
hold on;
surf(xs1, xs2, reshape(mu_2,Steps,Steps),'FaceAlpha','0.5');
