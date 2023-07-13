function w = weightCV(x,CV)
% x: raw data. In neural spike, x should be column in neuron and row in time
% CV: canonical variable, must in one vector

%SpikeOne = SpikeOne';
% 
% % Select the first column of 'Ud'
% Um_first_col = Um(:,1);
% 
% % Initialize R^2 vector
R2 = zeros(1, size(x, 2));
% 
% % Loop through each dimension in SpikeOne
for i = 1:size(x, 2)
%     
%     % Get the i-th dimension of SpikeOne
    x_i = x(:,i);
%     
%     % Create a linear regression model
    lm = fitlm(CV, x_i);
   
%     % Get the R^2 value for this dimension
    R2(i) = lm.Rsquared.Ordinary;
    
end
% 
% % Create a logical index of the R^2 values you're interested in
idx = ~isnan(R2) & ~isinf(R2) & R2 >= 0;
% 
% % Select only the R^2 values that meet your criteria
R2_selected = R2(idx);
% 
% % Calculate the mean of the selected R^2 values
R2_avg = mean(R2_selected);
% 
% % Print the mean R^2
w = R2_avg;
