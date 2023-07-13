function [PrincipalComponents,weightedPrincipalComponents, ProjectedData,...
    SingularValues,explained_variance_ratio] = svdPCA(x)
% This function is to calculate the principal component WITHOUT remove the
% mean of the input dataset. The first PC (PrincipalComponents) will be the
% mean of the input dataset

% x: input dataset. By default, x should be column in variable, row in observation

% PrincipalComponents: PC, the loading of PCA, the direction of projection

% ProjectedData: original dataset x projected to PC

% SingularValues: eigenvalue of covariance matrix of input dataset x, or latent

% explained_variance_ratio: percentage of variance explained by the PC, ranged from 0 to 1

[U, S, V] = svd(x);

SingularValues = diag(S);
singularVectors = U;
PrincipalComponents = V;% V is PC we want!
ProjectedData = x * V;
% 
% explained variance with:
variances = diag(S).^2 / (size(x, 1) - 1);

% Calculate the total variance
total_variance = sum(variances);

% Calculate the explained variance (sum = 1)
explained_variance_ratio = variances / total_variance;

% Convert to percentage (sum = 100,or 100%)
%explained_variance_percentage = 100 * explained_variance_ratio;

% Calculate the weighted principal components using the sqrt of variance as weight
weightedPrincipalComponents = PrincipalComponents(:,1:length(explained_variance_ratio)).*...
    sqrt(explained_variance_ratio)';

