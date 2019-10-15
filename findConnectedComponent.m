function [Hk,cc] = findConnectedComponent(H)
% Copyright (c) 2019 Qingzhe Li
% George Mason University
% qli10@gmu.edu; lzhao9@gmu.edu

% All right reserved. 

%{ 
The implmentation of Algorithm 2 FCD Heterogeneous Hypergraph Decomposition
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
% the index in featureList = cc{:,2} has counted the first feature.

[m,n] = size(H);
% m: number of components
% n: number of features
cc = cell(m+1,2);  % cc: connected conponents, the first column is the list of FCCs and the second column is the list of Features. 
mapFcc2CC = zeros(m,1);
exploredFeature = zeros(n,1);
% exploredFcc = zeros(m,1)
% exploredFcc = zeros(m,1)
% symetricH = H;
% islands = zeros(m,m);

[fccFeature,featureFCC] = toList(H);
% colLists = adjMatrix2Linkedlists(H,0);
ccCount = 0;
for row = 1:m
     if mapFcc2CC(row)==0
       ccCount = ccCount+1;
%        mapFcc2CC(row) = ccCount;
       dfs(row,ccCount);
     end
end

    function dfs(r,ccId)
      if mapFcc2CC(r)==0
        mapFcc2CC(r) = ccId;
        cc{ccId,1} = [cc{ccId,1},r];
%        islands(ccCount,row) = 1;
       curFeatureList = fccFeature{r};
       for i = 1:length(curFeatureList)
           curFeature = curFeatureList(i);           
           if exploredFeature(curFeature)==0
               exploredFeature(curFeature) = 1;
               curFccList = featureFCC{curFeature};
               for j = 1:length(curFccList)
                   curFcc = curFccList(j);
                   if mapFcc2CC(curFcc)==0
                        dfs(curFcc,ccId);
                   end;
               end
               
           end
       end
       
      end
    end

% islandFeature = cell(ccCount,1);
% for idx=1:ccCount
%    islandFeature{idx} = find(islands(idx,:)~=0);
% end
% featureIsland = mapFcc2CC;
Hk = cell(ccCount,1);
for k=1:ccCount
%     k
    cc{k,1} = sort(cc{k,1});
    fccList = cc{k,1};
%     s = sum(H(fccList,:),1)
    featureList = find(sum(H(fccList,:),1)~=0);
    
    cc{k,2} = [featureList+1,featureList+2+n];
    Hk{k} = H(fccList,featureList);
end
cc{ccCount+1,2} = [1,2+n];
cc = cc(1:ccCount+1,:);
end   % end main function


%% helper function
% function dfs(row,islandId)
% % global feature_island islands rowLists
% %     global feature_island
% %     global islands
% %     global rowLists
%     mapFcc2CC(row) = islandId;
%     islands(islandId,row) = 1;
%     lst = fccFeature{row}
%     for i=1:length(lst)
%         i
%         lst
%         lst(i)
%         size(mapFcc2CC)
%        if mapFcc2CC(lst(i))==0
%            mapFcc2CC(lst(i)) = islandId;
%            islands(islandId,lst(i)) = 1;
%            dfs(lst(i),islandId);
%        end
%     end
%     
% end

function [fccFeature,featureFcc] = toList(H)
    fccFeature = adjMatrix2Linkedlists(H,1);
    featureFcc = adjMatrix2Linkedlists(H,0);
end
function linkedLists = adjMatrix2Linkedlists(H,isRow)
    if ~isRow
        H = H';
    end
    [m,n] = size(H);
    linkedLists = cell(m,1);
    for r = 1:m
        lst = zeros(1,n);
        i = 0;
        for c = 1:n
            if H(r,c)~=0
                i = i+1;
                lst(i) = c; 
            end
        end
        linkedLists{r} = lst(lst~=0);
    end
end

