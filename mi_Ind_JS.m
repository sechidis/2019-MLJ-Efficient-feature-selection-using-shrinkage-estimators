function MI= mi_Ind_JS( x, y, arity_x, arity_y)

%%% Calculate the joint distributions for x and y 
arity_x = prod(arity_x);
[~,~,x] = unique(x,'rows');

arity_y = prod(arity_y);
[~,~,y] = unique(y,'rows');

n = length(y);
table_dim = [arity_x arity_y ];


%%% Find probabilities
pXY = accumarray([x y ],1,table_dim)/n;
pX  = sum(pXY,2); % arity_x * 1
pY = sum(pXY,1);  % 1 * arity_y

%%% Prepare some useful terms
pXY_ind = pX * pY;

pY_matrix = repmat(pX, 1, arity_y);
pX_matrix = repmat(pY, arity_x, 1);

pXY_sum = pX_matrix + pY_matrix;

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
exp_IND_square =  (n-1)*(n-2)*(n-3)*(pXY_ind.^2 + ... 
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

epsilon=10^(-50);
MI =sum(sum( pXY_shrink .* log(epsilon + pXY_shrink ./ (epsilon + sum(pXY_shrink,2) * sum(pXY_shrink,1)))));



