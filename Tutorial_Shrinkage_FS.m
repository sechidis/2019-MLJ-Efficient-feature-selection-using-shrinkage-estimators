% Load one of the provided dataset, e.g.
load('./Datasets/krvskp.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Estimate the arities of the features/class label %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
arities=[];
for feat = 1:size(data,2)
   arities(feat) =length(unique(data(:,feat)));
end
arities(size(data,2)+1)=length(unique(labels));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Select the features using our suggested algorithms %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we will select the topK features with our high-order criteria
% JMIplus and CMIMplus 
topK=10;
Selected_with_JMIplus = JMIplus(data,labels, topK, arities);
disp('Selected features using JMI+:')
disp(Selected_with_JMIplus)
Selected_with_CMIMplus = CMIMplus(data,labels, topK, arities);
disp('Selected features using CMIM+:')
disp(Selected_with_CMIMplus)