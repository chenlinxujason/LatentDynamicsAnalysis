%function [M_A,M_B,L_A_tr,L_B_tr,s_before,s_after,L_B_align] = CCA(L_A,L_B,m)
function [M_A,M_B,L_A_tr,L_B_tr] = CCA(L_A,L_B,m)
% L_A_tr: U
% L_B_tr: V
% m: how much PC for comparison

      % direct QR decompose 
%     [Q_A, R_A] = qr(L_A);
%     [Q_B, R_B] = qr(L_B);

%     % IncrementalQR, decomposing the matrix column by column
    [Q_A, R_A] =  IncrementalQR(L_A);
    [Q_B, R_B] =  IncrementalQR(L_B);

    Q_A = Q_A(:,1:m);
    R_A = R_A(1:m,1:m);
    Q_B = Q_B(:,1:m);
    R_B = R_B(1:m,1:m);
    Z = Q_A'*Q_B;

    [U,S,V] = svd(Z);

%    % 'inv' function may be memory and computationally intensive
%     M_A = inv(R_A)*U;
%     M_B = inv(R_B)*V;
    M_A = R_A\U; % backslash operator is more efficient
    M_B = R_B\V; % backslash operator is more efficient than inv

    L_A_tr = L_A(:,1:m)*M_A;%original
    L_B_tr = L_B(:,1:m)*M_B;%original

%     L_A_r = reshape(L_A,[],1);
%     L_B_r = reshape(L_B,[],1);
%     s_before = corrcoef(L_A_r,L_B_r);
%     s_before = s_before(1,2);
% 
%     L_A_tr_r = reshape(L_A_tr,[],1);
%     L_B_tr_r = reshape(L_B_tr,[],1);
%     s = corrcoef(L_A_tr_r,L_B_tr_r);
%     s = s(1,2);
% 
%     corr_coeffs = zeros(1, m);
%     for i = 1:m
%         corr_coeff = corrcoef(L_A(:, i), L_B(:, i));
%         corr_coeffs(i) = corr_coeff(1,2);
%         corr_coeff_tr = corrcoef(L_A_tr(:, i), L_B_tr(:, i));
%         corr_coeffs_tr(i) = corr_coeff_tr(1,2);
%     end
%     
    %s_before = corr_coeffs;
    %s_after = corr_coeffs_tr;

    %% For decoder input
    %L_A_align = (L_A(:,1:m)*M_A) / M_B; 
    %L_B_align = (L_B(:,1:m)*M_B) / M_A; 
end