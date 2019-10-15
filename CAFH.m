function [B,M,time_ADMM]=CAFH(data,D,H,lambda,power)
% Copyright (c) 2019  Liang Zhao
% George Mason University
% lzhao9@gmu.edu

% All right reserved. 
%{ 
The main function of the parameter optimization algorithm for the model CAFH
the detailed algorithmatic introduction is detailed in the paper

Notations:
    - p: original number of features; p+1: the number of features including dummy feature
    - q: number of feature components
    - n: number of observations

Input:
    - data:     n*(p+1)         training data sample matrix.
    - D:        q*1         feature component generation computation runtime
    - H:        q*(p+1)         incidence matrix denoting the correlation between feature and feature component: 
    - lambda:   scalar      any real number (default: 0.01)
    - power:    scalar      1/2 or 2/3.
Output: 
    - B:        2(p+1)*1        auxiliary value for getting feature weight, see details in Eqn (10).
    - M:        q*1         auxiliary value, see details in Eqn (9).
%} 

%% initialization
numObs = size(data,1);
X0 = data(:,1:end-1);
X_scale = max(abs(X0),[],1);
X0 = X0./repmat(X_scale,numObs,1);
X = [ones(numObs,1),X0];
Y = data(:,end);
Y(Y==0)=-1;
numFeatures = size(X,2);
numBasics = size(H,1);
H = [zeros(numBasics,1),H];
B = zeros(2*numFeatures,1);
Lamb = zeros(numBasics,1);
Omega1 = [eye(numFeatures),-eye(numFeatures)];
Omega2 = [eye(numFeatures),eye(numFeatures)];
ITERMAX = 20;
XO = X*Omega1;
HO = H*Omega2;
rho = 1;
% rho = 0.01;
% ERR = 0.0001;
ERR = 0.01;
% lambdaVect = lambda*[0;D];
D_init = D((numBasics-numFeatures+2):end);
lambdaVect = lambda*[0;D_init];

% initialize feature weight using reweighted-L1 logistic regression
W0 = run_model(X,Y,lambdaVect);
B(1:numFeatures,:)=max(W0,0);
B(numFeatures+1:end,:) = max(-W0,0);
M = H*Omega2*B;
fprintf('CAFH initialization\n');

tic
%% ADMM iterations
for i=1:ITERMAX
    B_old = B; M_old = M;
    
    %% subproblem for B update.
    B = update_B(B,Y,M,XO,HO,Lamb,rho);
    
    %% subproblem for M update.
    M = update_M(HO,B,Lamb,rho,D,lambda,power);
    
    %% update dual variable.
    Lamb = Lamb + (M-HO*B);
    
    %% calcualte primal and dual residuals
    p = norm(M-HO*B,'fro');
    d = rho*norm(HO'*(M-M_old),'fro');
    fprintf('%d\tp:%f\td:%f\t%f\n',i,p,d,rho);
    
    %% update rho (can be commented out based on convergence performance)
    if(p>10*d)
        rho = 2*rho;
    else
        if(10*p<d)
            rho = rho/2;
        end
    end
    %
    %% termination criterion
    if p < ERR && d < ERR
        break;
    end
end
time_ADMM = toc;
end

function M = update_M(P,B,Lamb,rho,D,lambda,p)
%% Analytical solution to the subproblem of M update. 

numBasics = size(P,1);
M = [];
for i=1:numBasics
    
    if p==1/2 % see Eqn (16) and (17)
        r = roots([1,0,-(P(i,:)*B-Lamb(i,1)),lambda/(2*rho)*D(i,1)]);
        rr = r([isreal(r(1,1)),isreal(r(2,1)),isreal(r(3,1))]);
        M = [M;max(max(rr),0)^2];
        
    elseif p == 2/3 % see Eqn (18) and (19)
        r = roots([1,0,0,-(P(i,:)*B-Lamb(i,1)),2*lambda/(3*rho)*D(i,1)]);
        rr = r([isreal(r(1,1)),isreal(r(2,1)),isreal(r(3,1)),isreal(r(4,1))]);
        if isempty(rr)
            M = [M;0];
        else
            M = [M;max(max(rr),0)^3];
        end
    end
end
end

function B=update_B(B0,Y,M,X,P,Lamb,rho)
%% Analytical solution to the subproblem of B update. 

numFeatures = size(B0,1);
numObs = size(Y,1);
y_B = 10e10;
MAX_ITER = 1000;
B = B0;

function res = y(b)
XBY = X*b.*Y;
res1 = zeros(size(XBY));
res1(XBY>=0,:)=log(1 + exp(-XBY(XBY>=0,:)))/numObs;
res1(XBY<0,:)=(log(exp(XBY(XBY<0,:))+1)-XBY(XBY<0,:))/numObs;
res = sum(res1)+(rho/2)*sum(sum((P*b-M-Lamb).^2));
end

beta = 1;
break_flag = 0;
for iter = 1:MAX_ITER
    %% calculate gradient
    alpha = 1;
    ratio = 0.5;
    gradient = -(sum((X.*repmat(Y,1,numFeatures))./repmat(1+exp((X*B).*Y),...
                        1,numFeatures),1)/numObs)'+rho*((P')*P*B-P'*(M+Lamb));
    y_B0 = y(B);
    B0 = B;
    
    %% backtracking Armijo line search
    while true
        B = B0-alpha*gradient;
        B = max(B,0);
        p = B-B0;
        r_norm = norm(p,'fro');

        if(r_norm<1e-4)
            break_flag = 1;
            break;
        end
        y_B = y(B);
        if y_B<y_B0+alpha*beta*p'*gradient
            break;
        end
        alpha = alpha * ratio;
    end
    
    %% termination criterion
    if break_flag == 1
        break_flag = 0;
        break;
    end
end
end
function [wSPG] = run_model(X_tr,Y_tr,lambdaVect)
%% Reweighted-L1 logistic regression
% Based on spectral gradient descent
%{
Output:
    - wSPG: (p+1)*1 feature weights estimated.
%}
numFeatures = size(X_tr,2)-1;
w_init = zeros(numFeatures+1,1);
funObj = @(w)LogisticLoss(w,X_tr,Y_tr);

%% Set Optimization Options
gOptions.maxIter = 2000;
gOptions.verbose = 0; % Set to 0 to turn off output
options.corrections = 10; % Number of corrections to store for L-BFGS methods

%% Run Solvers
options = gOptions;
wSPG = L1General2_SPG(funObj,w_init,lambdaVect,options);
end
