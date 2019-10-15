function [confusionMats]=get_predict_result(test,Ws,lambdas)
numObs_te = size(test,1);
numFeatures = size(test,2)-1;
numLambdas = size(Ws,2);
X_te = [ones(numObs_te,1) test(:,1:numFeatures)];
tmp_val = X_te./repmat(max(abs(X_te),[],1),numObs_te,1);
X_te(:,sum(abs(X_te))~=0) = tmp_val(:,sum(abs(X_te))~=0);
Y_te = test(:,end);
Y_te(Y_te==0,1)=-1;
confusionMats = {};
for i=1:numLambdas
    W = Ws(:,i);
    Y_pred = sign(X_te * W);
    mat.tp = sum((Y_pred == 1) & (Y_te == 1));
    mat.fp = sum((Y_pred == 1) & (Y_te ~= 1));
    mat.tn = sum((Y_pred ~= 1) & (Y_te ~= 1));
    mat.fn = sum((Y_pred ~= 1) & (Y_te == 1));
    mat.lambda = lambdas(1,i);
    confusionMats = {confusionMats{:},mat};
end
end