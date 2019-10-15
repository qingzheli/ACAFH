function [Ws,cMats,accs,runtimes,K,time_ADMM] = run_ACAFH(data_tr,data_te,H,D,lambdas,pow)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main function of Acclerated Cost-Aware classification using the FCD Heterogeneous hypergraph (ACAFH)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Copyright (c) 2019 Qingzhe Li
% George Mason University
% qli10@gmu.edu; 

% All right reserved. 


% load data and FCD Heterogeneous hypergraph, data format will be introduced
% in the function ACAFH.

[Hk,connectedComponent] = findConnectedComponent(H);
K = length(Hk);

% feature weights. 
Ws = []; 

% choose the value of l_p quasi-norm (0<p<1), when the power = 1/2, it means
% l_{1/2} quasi-norm. When the power = 2/3, it means l_{1/2} quasi-norm.
% power can be only 1/2 or 2/3, which ensures analytical solutions.
% power = 1/2; 
% power = 2/3;
power = pow;

for lambda=lambdas
    % call the ADMM-based algorithm.
    [B,M,time_ADMM] = ACAFH(data_tr,D,Hk,lambda,power,connectedComponent);

    numFeatures = size(B,1)/2;
    
    W = (B'*[eye(numFeatures);-eye(numFeatures)])'; 
    Ws = [Ws,W];
end
% calculate the confusion matrix.
cMats = get_predict_result(data_te,Ws,lambdas);

% calculate the accuracy and runtime.
[accs,runtimes] = postprocess(Ws,cMats,D,H);
end

function [results,runtimes] = postprocess(ws,cmats,D,H)
ws = ws(2:end,:);
runtimes = [];
for i=1:size(ws,2)
    ws_basic = H*ws(:,i);
    % estimate the feature generation runtime.
    runtimes = [runtimes,sum(D(ws_basic(:,1)~=0,:))];
end
% get the precision, recall, and F-measure.
results = zeros(0,3);
for c=cmats
    c = c{1,1};
    p = c.tp/(c.tp+c.fp);
    r = c.tp/(c.tp+c.fn);
    f = 2*p*r/(p+r);
    results(end+1,:)=[p,r,f];
end
end