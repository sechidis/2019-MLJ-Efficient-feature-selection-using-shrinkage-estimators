% Load one of the provided dataset, e.g.
load('./Datasets/krvskp.mat')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Estimate the arities of the features/class label %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
arities=[];
for feat = 1:size(X_data,2)
   arities(feat) =length(unique(X_data(:,feat)));
end
arities(size(X_data,2)+1)=length(unique(Y_labels));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Select the features using our suggested algorithms %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we will select the topK features with our four high-order criteria
% JMI-3, JMI-4, CMIM-3 and CMIM-4
topK=10;
Selected_with_JMI3 = JMI3(X_data,Y_labels, topK, arities);
disp('Selected features using JMI-3:')
disp(Selected_with_JMI3)
Selected_with_JMI4 = JMI4(X_data,Y_labels, topK, arities);
disp('Selected features using JMI-4:')
disp(Selected_with_JMI4)
Selected_with_CMIM3 = CMIM3(X_data,Y_labels, topK, arities);
disp('Selected features using CMIM-3:')
disp(Selected_with_CMIM3)
Selected_with_CMIM4 = CMIM4(X_data,Y_labels, topK, arities);
disp('Selected features using CMIM-4:')
disp(Selected_with_CMIM4)