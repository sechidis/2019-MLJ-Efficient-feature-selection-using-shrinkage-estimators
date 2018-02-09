function [selectedFeatures] = JMIplus(data,labels, topK, arities)
% Summary
%    JMIplus algorithm for feature selection
% Inputs
%    data: n x d matrix X, with categorical values for n examples and d features
%    labels: n x 1 vector with the labels
%    topK: Number of features to be selected
%    arities: (d+1)x1 vector, that in the first d positions are the arities
%                of the feature, and in the last of the label 

numFeatures = size(data,2);

%%%%%%%%%% First feature
count = 1;
mi_score = zeros(1,numFeatures);

for index_feature = 1:numFeatures
    score_per_feature(index_feature) =  mi_Ind_JS(data(:,index_feature),labels,arities(index_feature),arities(end));
end
[val_max,selectedFeatures(1)]= max(score_per_feature);
mi_score(selectedFeatures(1)) = val_max;
not_selected_features = setdiff(1:numFeatures,selectedFeatures);

%%%%%%%%%% Second feature
count = 2;
score_per_feature = zeros(1,numFeatures);
score_per_feature(selectedFeatures(1)) = NaN;

for index_feature_ns = 1:length(not_selected_features)
    
    score_per_feature(not_selected_features(index_feature_ns)) = score_per_feature(not_selected_features(index_feature_ns))+mi_Ind_JS([data(:,not_selected_features(index_feature_ns)),data(:, selectedFeatures(count-1))], labels,arities([not_selected_features(index_feature_ns) selectedFeatures(count-1)]),arities(end));
    
end

[val_max,selectedFeatures(count)]= nanmax(score_per_feature);


score_per_feature(selectedFeatures(count)) = NaN;
not_selected_features = setdiff(1:numFeatures,selectedFeatures);


%%%%%%%%%% Rest of the features

%%% Efficient implementation of the second step, at this point I will store
%%% the score of each feature. Whenever I select a feature I put NaN score

score_per_feature = zeros(1,numFeatures);
score_per_feature(selectedFeatures(1:2)) = NaN;
count = 3;
while count<=topK
    
    for index_feature_ns = 1:length(not_selected_features)
        
        
        for index_feature_s = 1:(length(selectedFeatures)-1)
            
            score_per_feature(not_selected_features(index_feature_ns)) = score_per_feature(not_selected_features(index_feature_ns))+mi_Ind_JS([data(:,not_selected_features(index_feature_ns)),data(:, selectedFeatures(count-1)),data(:, selectedFeatures(index_feature_s))], labels,arities([not_selected_features(index_feature_ns) selectedFeatures(count-1) selectedFeatures(index_feature_s)]),arities(end));
        end
        
    end
    
    [val_max,selectedFeatures(count)]= nanmax(score_per_feature);
    
    
    score_per_feature(selectedFeatures(count)) = NaN;
    not_selected_features = setdiff(1:numFeatures,selectedFeatures);
    count = count+1;
end