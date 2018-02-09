function CMI= cmi_Ind_JS( x, y, z, arity_x, arity_y, arity_z)


%%% Calculate the joint distributions for x, y  and z
arity_x = prod(arity_x);
[~,~,x] = unique(x,'rows');

arity_y = prod(arity_y);
[~,~,y] = unique(y,'rows');

arity_z = prod(arity_z);
[~,~,z] = unique(z,'rows');


n = length(y);
table_dim = [arity_x arity_y arity_z];

%%% Find probabilities
 pXYZ_3d = accumarray([x y z],1,table_dim)/n;
 for y_index = 1:arity_y
 p_XZ_2d = squeeze(pXYZ_3d(:,y_index,:));
 pXY(:,y_index)= p_XZ_2d(:);
 end
 
 %%%%%%%%%% Here the same code as for MI
pX  = sum(pXY,2); % arity_x * 1  
pY = sum(pXY,1);  % 1 * arity_y

%%% Prepare some useful terms
pXY_ind = pX * pY; % arity_x * arity_y

pY_matrix = repmat(pX, 1, arity_y);
pX_matrix = repmat(pY, arity_x*arity_z, 1);

pXY_sum = pX_matrix + pY_matrix; % arity_x * arity_y

%%% Estimate the five terms
%%% Term 1
var_ML = pXY.*(1-pXY)/n;

%%% Term 2
 cov_MLvsIND = (pXY/n^2) .* ( (n-1)*(pXY_sum - 2*pXY_ind) + 1 - pXY );

%%% Term 3
exp_ML_square = ((n-1) * (pXY.^2) + pXY)/n;

%%% Term 5
exp_MLvsIND =  pXY .* ( (n^2-3*n+2)*pXY_ind + (n-1)* (pXY_sum+pXY ) + 1)/n^2;

%%% Term 4
exp_IND_square =  (n-1)*(n-2)*(n-3)*(pXY_ind.^2 + 4*(pX_matrix - pXY).* (pXY.^2) .*(pY_matrix - pXY) ) + ...
                (n-1)*(n-2)*pXY_ind.*(pXY_sum + 4*pXY) + ...
                (n-1)*(2*pXY .* pXY_sum + 2*pXY.^2 + pXY_ind) + ...
                pXY;
            
exp_IND_square = (exp_IND_square/n^3);



% Estimate numerator/denominator
numerator = sum(sum(var_ML-cov_MLvsIND));
denominator = sum(sum(exp_ML_square + exp_IND_square -  2*exp_MLvsIND));

lambda_opt = max(0,min(1, numerator/denominator ));

% Estimate shrinkage probabilities
pXY_shrink = lambda_opt * pXY_ind + (1-lambda_opt) * pXY;
  for y_index = 1:arity_y
      pXYZ_shrink(:,y_index,:) = reshape(pXY_shrink(:,y_index),arity_x,arity_z);
  end
 
  epsilon=10^(-50);
CMI=0;
for zIndex = 1:arity_z
CMI = CMI+ sum(sum( pXYZ_shrink(:,:,zIndex) .* log(epsilon + sum(sum(pXYZ_shrink(:,:,zIndex)))*pXYZ_shrink(:,:,zIndex) ./ (epsilon + sum(pXYZ_shrink(:,:,zIndex),2) * sum(pXYZ_shrink(:,:,zIndex),1)))));
end

