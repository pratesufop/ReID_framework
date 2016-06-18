function [Beta,names] = thresholdBetaScore(r, k, n, sigma)
%THRESHOLDBETASCORE Compute p-values based on Beta distribution
%   [Beta,names] = THRESHOLDBETASCORE(r, k, n, sigma) 
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   Copyright (2013) Nejc Ilc <nejc.ilc@gmail.com> 
%   Based on R package RobustRankAggreg written by Raivo Kolde. 
%   Reference:
%     Kolde, R., Laur, S., Adler, P., & Vilo, J. (2012).
%     Robust rank aggregation for gene list integration and meta-analysis.
%     Bioinformatics, 28(4), 573-580
%   
%   Revision: 1.0 Date: 2013/05/16
%--------------------------------------------------------------------------

% Input variables validation
    rLen = length(r);

    if ~exist('k','var') || isempty(k)
        k = 1:rLen;
    end
    if ~exist('n','var') || isempty(n)
        n = rLen;
    end
    if ~exist('sigma','var') || isempty(sigma)
        sigma = ones(1,n);
    end

    if(length(sigma) ~= n)
        error('The length of sigma does not match n!');
    end
    if(length(r) ~= n)
        error('The length of p-values does not match n!');
    end
    if(min(sigma)< 0 || max(sigma) > 1)
        error('Elements of sigma are not in the range [0,1]!');
    end
    if(any(~isnan(r) & r > sigma))
        error('Elements of r must be smaller than elements of sigma!');
    end
%--------------------------------------------------------------------------

    x = sort(r(~isnan(r)));
    sigma = sort(sigma, 'descend');
    Beta = nan(1, length(k));

    for i = 1:length(k)

        if(k(i) > n)
            Beta(i) = 0;
            continue;
        end
        if(k(i) > length(x))
            Beta(i) = 1;
            continue;
        end
        if(sigma(n) >= x(k(i)))
            Beta(i) = betacdf( x(k(i)), k(i), n + 1 - k(i));
            continue;
        end
        
        % Non-trivial cases
        % Find the last element such that sigma(n0) <= x(k(i))
        n0 = find(sigma < x(k(i)));
        n0 = n0(1) - 1;

        % Compute beta score vector beta(n,k) for n = n0 and k = 1..k(i)
        if(n0 == 0)
            B = [1, zeros(1, k(i))];
        elseif(k(i) > n0)
            B = [1, betacdf(x(k(i)), 1:n0, n0:-1:1), zeros(1, k(i) - n0)];
        else
            B = [1, betacdf(x(k(i)), 1:k(i), n0+1-(1:k(i)) )];
        end

        % In the following update steps sigma < x(k(i))
        z = sigma( (n0+1) : n );
        for j = 1:(n - n0)
            B( 2:(k(i)+1)) = (1-z(j)) * B(2:(k(i)+1)) + z(j) * B(1:k(i));
        end

        Beta(i) = B(k(i)+1);
    end

    names = k;

end