function pval = correctBetaPvalues(p,k)
%CORRECTBETAPVALUES Compute p-values based on Beta distribution
%   pval = CORRECTBETAPVALUES(p,k) 
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
    pval = betacdf(p,1,k);
end