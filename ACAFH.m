
function [B,M,time_ADMM]=ACAFH(data,D,Hk,lambda,power,connectedComponent)
% Copyright (c) 2019 Qingzhe Li
% George Mason University
% qli10@gmu.edu
%{ 
 
The implmentation of the parameter optimization algorithm for the model
ACAFH (Algorithm 2)
the detailed algorithmatic introduction is detailed in the paper

Notations:
    - p: original number of features; p+1: the number of features including dummy feature
    - q: number of feature components
    - n: number of observations

Input:
    - data:     n*(p+1)         training data sample matrix.
    - D:        q*1         feature component generation computation runtime
    - Hk:        q*(p+1)         incidence matrix of the k-th CCHH denoting the correlation between feature and feature component: 
    - lambda:   scalar      any real number (default: 0.01)
    - power:    scalar      1/2 or 2/3.
    - connectedComponent:   the first column is the list of FCCs and the second column is the list of Features. 
Output: 
    - B:        2(p+1)*1        auxiliary value for getting feature weight, see details in Eqn (10).
    - M:        q*1         auxiliary value, see details in Eqn (9).
%} 

%% initialization
K = length(Hk); %fcc Count
KaddOne = size(connectedComponent,1);
fccLists = connectedComponent(:,1);
featureLists = connectedComponent(:,2);

numObs = size(data,1);
X0 = data(:,1:end-1);
X_scale = max(abs(X0),[],1);
X0 = X0./repmat(X_scale,numObs,1);  % -1 ~ 1 normalized data for each column/feature
X = [ones(numObs,1),X0];
Y = data(:,end);
Y(Y==0)=-1;
numFeatures = size(X,2);
numBasics = size(D,1); % Count of FCCs
HO2k = cell(K,1);
Omega1k = cell(K,1);
Omega2k = cell(K,1);
Bk = cell(KaddOne,1);
for k = 1:K   
   Omega1k{k} = [eye(size(Hk{k},2)),-eye(size(Hk{k},2))];
   Omega2k{k} = [eye(size(Hk{k},2)),eye(size(Hk{k},2))];
   HO2k{k} = Hk{k}*Omega2k{k};
end

B = zeros(2*numFeatures,1);
Lamb = zeros(numBasics,1);
HhatOB = zeros(numBasics,1);  % store the intermediate result of H*\hat{Omega}*B


ITERMAX = 20;

XO = [X,-X];
rho = 1;
ERR = 0.01;
D_init = D((numBasics-numFeatures+2):end);
lambdaVect = lambda*[0;D_init];
% initialize feature weight using reweighted-L1 logistic regression
W0 = run_model(X,Y,lambdaVect);
B(1:numFeatures,:)=max(W0,0);
B(numFeatures+1:end,:) = max(-W0,0);
M = zeros(numBasics,1);
for k = 1:K
    Bk{k} = B(featureLists{k},1);
   M(fccLists{k},:) = HO2k{k}*Bk{k};
end

Bk{KaddOne} = B(featureLists{KaddOne},1);

fprintf('ACAFH initialization: ADMM_ITER = %d  \n lambda=%4.4f\tpower=%4.4f\trho=%4.4f\tERR=%4.4f K=%d\n',ITERMAX,lambda,power,rho,ERR,K);

%% ADMM iterations
tic
for i=1:ITERMAX
%     B_old = B;
%     Bk_old=Bk;
    M_old = M; 
    
    %% subproblem for B update.
    [B,Bk,HhatOB] = update_B(B,Bk,Y,M,XO,HO2k,Lamb,rho,fccLists,featureLists);
    
    %% subproblem for M update.
    M = update_M(HhatOB,Lamb,rho,D,lambda,power);
    
    %% update dual variable.
    Lamb = Lamb + (M-HhatOB);
    
    %% calcualte primal and dual residuals
    
    p = norm(M-HhatOB,'fro');
    sqrSumHO_M_Mold = 0;
    for k = 1:K
        sqrSumHO_M_Mold = sqrSumHO_M_Mold+sum((HO2k{k,1}*(M(fccLists{k},1)-M_old(fccLists{k},1))).^2);
    end;
%     d = rho*norm(HO'*(M-M_old),'fro');
    d = rho*sqrt(sqrSumHO_M_Mold);
    fprintf('%d\tp:%f\td:%f\t%f\t',i,p,d,rho);
    
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
end  %% end of main function

function M = update_M(HhatOB,Lamb,rho,D,lambda,p)
%% Analytical solution to the subproblem of M update. 

numBasics = size(HhatOB,1);
M = [];
for i=1:numBasics
    
    if p==1/2 % see Eqn (32) and (33)
        r = roots([1,0,-(HhatOB(i)-Lamb(i,1)),lambda/(2*rho)*D(i,1)]);
        rr = r([isreal(r(1,1)),isreal(r(2,1)),isreal(r(3,1))]);
        M = [M;max(max(rr),0)^2];
        
    elseif p == 2/3 % see Eqn (34) and (35)
        r = roots([1,0,0,-(HhatOB(i)-Lamb(i,1)),2*lambda/(3*rho)*D(i,1)]);
        rr = r([isreal(r(1,1)),isreal(r(2,1)),isreal(r(3,1)),isreal(r(4,1))]);
        if isempty(rr)
            M = [M;0];
        else
            M = [M;max(max(rr),0)^3];
        end
    end
end
end

function [B,Bk,HhatOB]=update_B(B0,Bk,Y,M,X,Pk,Lamb,rho,fccLists,featureLists)
%% Analytical solution to the subproblem of B update.
% p: HO2k
K = size(Pk,1);
KaddOne = K+1;
numFeatures = size(B0,1);  
numObs = size(Y,1);
MAX_ITER = 1000;
B = B0;

%%
function res = y_ori(b)
for k = 1:KaddOne
%     k
    if k<KaddOne
    pb =  Pk{k}*Bk{k};
    HhatOB(fccLists{k}',1) = pb;
    end
%     XBY = XBY+X(:,featureLists{k})*b{k};
end
XBY = X*b.*Y;
res1 = zeros(size(XBY));
res1(XBY>=0,:)=log(1 + exp(-XBY(XBY>=0,:)))/numObs;
res1(XBY<0,:)=(log(exp(XBY(XBY<0,:))+1)-XBY(XBY<0,:))/numObs;
res = sum(res1)+(rho/2)*sum(sum((HhatOB-M-Lamb).^2));
% res = sum(res1)+(rho/2)*sum(sum((P*b-M-Lamb).^2));
end

beta = 1;
break_flag = 0;
for iter = 1:MAX_ITER
    %% calculate gradient by only using Hk to improve time complexity and memory consumption. 
    alpha = 1;
    ratio = 0.5;
    
    grad1 = -(sum((X.*repmat(Y,1,numFeatures))./repmat(1+exp((X*B).*Y),1,numFeatures),1)/numObs)'; 
    
    PPB = zeros(numFeatures,1);
    MaddLamb = M+Lamb;
    PMaddLamb = zeros(numFeatures,1);
    for k = 1:K
       fccList = fccLists{k};
       featureList = featureLists{k};

       idices = featureList;
        
       PPB(idices) = (Pk{k})'*Pk{k}*Bk{k};
       PMaddLamb(idices) = Pk{k}'* MaddLamb(fccList);
    end
    grad2 = rho*(PPB-PMaddLamb);
    gradient = grad1+grad2;

    y_B0 = y_ori(B);

    B0 = B;
    
     %% backtracking Armijo line search
    while true
        B = B0-alpha*gradient;
        B = max(B,0);
     % copy B to Bk
        for k = 1:KaddOne
            Bk{k} = B(featureLists{k});
        end
        p = B-B0;
        r_norm = norm(p,'fro');

        if(r_norm<1e-3)
            break_flag = 1;
            break;
        end
        y_B = y_ori(B);

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
fprintf('Break at iteration: %d\t%2.5f\n',iter,r_norm);
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
