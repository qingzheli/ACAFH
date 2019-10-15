% Copyright (c) 2019 Qingzhe Li; Liang Zhao
% George Mason University
% qli10@gmu.edu; lzhao9@gmu.edu

% Allright reserved. 
% ---------------
% ENVIRONMENT: 
% ---------------
% Matlab 2011-2019 (previous version might also work)
% 
% 
% ---------------
% INSTALLATION: 
% ---------------
% 1. Before running the codes, please first download Matlab package "L1General" from the link: http://www.cs.ubc.ca/~schmidtm/Software/L1General.zip
% 2. unzip the downloaded package
% 3. open Matlab, then set the package folder as the current path of Matlab.
% 4. add the package path using the following command
% >> addpath(genpath(pwd));
% 
% ---------------
% EXAMPLE RUN:
% ---------------
% 1. Reset the current path to the folder of our code.
% 2. Run the following command line:
% >> example_run
clear;
load('sampleData.mat');
lambdas = 2.^-20; 
powers = [1/2,2/3];
diff = zeros(1,6);
datasetCount = 1;
ithDataset = 1;
trainingTime = zeros(datasetCount,4);  % storing the training time of optimizing the ACAFH and CAFH models.
total_time = zeros(datasetCount,4);  % storing the total running time (training and testing) for both ACAFH and CAFH models.
accs = cell(datasetCount,4); % the prediction accuracy using the features selected by ACAFH and CAFH models.
Ws = cell(datasetCount,4); % the feature weights
runtimes = cell(datasetCount,4); % the prediction time using the features selected by ACAFH and CAFH models.
%% ACAFH: p = 1/2
tic
[Ws{ithDataset,1},cMat1,accs{ithDataset,1},runtimes{ithDataset,1},Ks(ithDataset),time_ADMM] = run_ACAFH(data{1},data{2},data{3},data{4},lambdas,powers(1));
trainingTime(ithDataset,1) = time_ADMM;
total_time(ithDataset,1) = toc;
%% CAFH: p = 1/2
tic
[Ws{ithDataset,2},cMat,accs{ithDataset,2},runtimes{ithDataset,2},time_ADMM] = run_CAFH(data{1},data{2},data{3},data{4},lambdas,powers(1));
total_time(ithDataset,2) = toc;
trainingTime(ithDataset,2) = time_ADMM;
%% ACAFH: p = 2/3
 tic
[Ws{ithDataset,3},cMat1,accs{ithDataset,3},runtimes{ithDataset,3},Ks(ithDataset),time_ADMM] = run_ACAFH(data{1},data{2},data{3},data{4},lambdas,powers(2));
trainingTime(ithDataset,3) = time_ADMM;

total_time(ithDataset,3) = toc;
%% CAFH: p = 2/3
tic
[Ws{ithDataset,4},cMat,accs{ithDataset,4},runtimes{ithDataset,4},time_ADMM] = run_CAFH(data{1},data{2},data{3},data{4},lambdas,powers(2));
total_time(ithDataset,4) = toc;
trainingTime(ithDataset,4) = time_ADMM;


diff(ithDataset,1) = norm(accs{ithDataset,1}-accs{ithDataset,2},'fro'); 
diff(ithDataset,2) = norm(Ws{ithDataset,1}-Ws{ithDataset,2},'fro');
diff(ithDataset,3) = norm(runtimes{ithDataset,1}-runtimes{ithDataset,2},'fro');
diff(ithDataset,4) = norm(accs{ithDataset,3}-accs{ithDataset,4},'fro'); 
diff(ithDataset,5) = norm(Ws{ithDataset,3}-Ws{ithDataset,4},'fro');
diff(ithDataset,6) = norm(runtimes{ithDataset,3}-runtimes{ithDataset,4},'fro');

disp('All done!');

disp('Training time:'); 
fprintf('ACAFH model (p=1/2): %d\n', trainingTime(1,1)); 
fprintf(' CAFH model (p=1/2): %d\n', trainingTime(1,2)); 
fprintf('ACAFH model (p=2/3): %d\n', trainingTime(1,3)); 
fprintf(' CAFH model (p=2/3): %d\n', trainingTime(1,4)); 