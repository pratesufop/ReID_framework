function aggR = stuart(rmat)
%STUART Compute aggregated rank with Stuart-Aerts method
%   aggR = STUART(rmat) 
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
%   INPUTS:
%       rmat    numeric matrix with as many rows as the number of
%               unique elements and with as many columns as the number 
%               of ranked lists. All entries have to be on interval 
%               [0,1]. Smaller numbers correspond to better ranking. 
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   OUTPUTS:
%       aggR    vector of aggregated ranks (p-values)
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   See also    AGGREGATERANKS.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   Copyright (2013) Nejc Ilc <nejc.ilc@gmail.com> 
%   Based on R package RobustRankAggreg written by Raivo Kolde. 
%   References:
%     Kolde, R., Laur, S., Adler, P., & Vilo, J. (2012). 
%     Robust rank aggregation for gene list integration and meta-analysis.
%     Bioinformatics, 28(4), 573-580
%
%     Stuart, J. M., Segal, E., Koller, D., & Kim, S. K. (2003). 
%     A gene-coexpression network for global discovery of conserved genetic
%     modules. Science, 302(5643), 249-55
%
%     Aerts, S., Lambrechts, D., Maity, S., Van Loo, P., Coessens, B., De
%     Smet, F., Tranchevent, L.-C., et al. (2006). Gene prioritization
%     through genomic data fusion. Nature biotechnology, 24(5), 537-44
%   
%   Revision: 1.0 Date: 2013/05/16
%--------------------------------------------------------------------------
	rmat = sort(rmat, 2);
    N=size(rmat,1);
	aggR = zeros(N,1);
    for ai = 1:N
       aggR(ai) = qStuart(rmat(ai,:)); 
    end
end

% Stuart-Aerts method helper function
function q=qStuart(r)
	N = sum(~isnan(r));
	v = ones(1, N+1);
    for k = 1:N
        v(k+1) = sumStuart( v(1:k), r(N-k+1));
    end
	q = factorial(N) * v(N+1);
end

% Stuart-Aerts method helper functions
function s = sumStuart(v, r)
	k = length(v);
	l_k = 1:k;
	ones = (-1).^(l_k + 1);
	f = factorial(l_k);
	p = r.^l_k;
	s = ones * ( v(end:-1:1) .* p ./ f)';
end

