function [Q, R] = IncrementalQR(A)
    % Initialize Q and R
    [n, m] = size(A);
    Q = zeros(n, m);
    R = zeros(m, m);

    % Iterate over each column of A
    for k = 1:m
        % Compute the kth column of Q and the (k,k) element of R
        q = A(:, k);
        for i = 1:k-1
            R(i, k) = Q(:, i)'*A(:, k);
            q = q - R(i, k)*Q(:, i);
        end
        R(k, k) = norm(q);
        Q(:, k) = q / R(k, k);
    end
end