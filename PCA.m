function [coeff,projection,explained,m] = PCA(x,method,level_var_explained)

% x: Data matrix. Rows of x correspond to observations and columns correspond to variables

% method
%   - 1: Standard demean PCA; 
%   - 2: Add pc0 in the first column as the mean of data x  
%   - 3: No deman PCA, in this case the first column pc1 will be 
%        the direction of the mean of data x

% level_var_explained: how much variance need to be explained, recommend 0.9

% 'cov' function: if user already demean the data x, then MATLAV 'cov' will not
% substract the mean again. However, if user not demean the data x, the MATLAB 
% 'cov' will first susbract the mean then compute covariance matrix

  data = x;

switch method
         case 1 % Standard demean PCA (same as MATLAB method)
            data_mean = mean(data,1);
            centered_data = data - data_mean; 
            covariance_matrix = cov(centered_data); 
            [eigen_vectors, eigen_values] = eig(covariance_matrix);
            eigen_values = diag(eigen_values);
            [~, sort_indices] = sort(eigen_values, 'descend');
            eigen_values = eigen_values(sort_indices);
            eigen_vectors = eigen_vectors(:, sort_indices);

            coeff = eigen_vectors;
            projection = centered_data * coeff;
            explained = eigen_values/sum(eigen_values);

            explained_cum = cumsum(explained);
            [row, ~] = find(explained_cum > level_var_explained);
            m = row(1);
               
        case 2 % mean + standard PCA
            data_mean = mean(data,1);
            centered_data = data - data_mean;
            covariance_matrix = cov(centered_data);
            [eigen_vectors, eigen_values] = eig(covariance_matrix);
            eigen_values = diag(eigen_values);
            [~, sort_indices] = sort(eigen_values, 'descend');
            eigen_values = eigen_values(sort_indices);
            eigen_vectors = eigen_vectors(:, sort_indices);

            data_size = size(data);
            M = data_size(1);
            ssq_tot = sum(diag(data'*data));
            ssq_mean = M*data_mean*data_mean';
            ssq_err = ssq_tot - ssq_mean;
            ssq_pc0 = sqrt(sum(data_mean.^2));
            pc0 = data_mean / ssq_pc0;

            explained_ori = eigen_values/sum(eigen_values);
            ssq_pc = explained_ori*ssq_err;
            explained = ssq_pc / ssq_tot;
            explained_pc0 = ssq_mean / ssq_tot;

            coeff = [pc0',eigen_vectors];
            projection = data * coeff;
            explained = [explained_pc0;explained];

            explained_cum = cumsum(explained);
            [row, ~] = find(explained_cum > level_var_explained);
            m = row(1);
            
        case 3% NO demean method - need to manually calculate covariance matrix
            centered_data = data;
            covariance_matrix = centered_data'*centered_data/(size(centered_data,1) - 1);
            %covariance_matrix = cov(centered_data);
            [eigen_vectors, eigen_values] = eig(covariance_matrix);
            eigen_values = diag(eigen_values);
            [~, sort_indices] = sort(eigen_values, 'descend');
            eigen_values = eigen_values(sort_indices);
            eigen_vectors = eigen_vectors(:, sort_indices);

            coeff = eigen_vectors;
            projection = centered_data * coeff;
            explained = eigen_values/sum(eigen_values);

            explained_cum = cumsum(explained);
            [row, ~] = find(explained_cum > level_var_explained);
            m = row(1);                
end