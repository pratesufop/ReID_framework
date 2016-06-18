function rho = rhoScores(r, topCutoff)
%RHOSCORES Compute Rho scores for rank vector
%   rho = RHOSCORES(r, topCutoff) 
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
%   INPUTS:
%       r           vector of normalized rank values on interval [0,1]
%       topCutoff   a vector of cutoff values used to limit the number of 
%                   elements in the input lists
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   OUTPUTS:
%       rho         a vector of rho values, corrected against bias from
%                   multiple hypothesis testing (Bonferroni correction).
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   See also CORRECTBETAPVALUES, THRESHOLDBETASCORE, AGGREGATERANKS.
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
    if ~exist('topCutoff','var') || isempty(topCutoff)
        topCutoff = NaN;
    end
    
    if isvector(r)
        rows = 1;
        r = r(:)'; % force row vector form
    else
        rows = size(r,1);
    end
    
    rho = nan(rows,1);
    
    for rInd = 1:rows
        r1 = r(rInd,:);
        
        if(isnan(topCutoff(1)))
            x = betaScores(r1);
            % Correct using Bonferroni method.
            rho(rInd) = correctBetaPvalues( min(x), sum(~isnan(x)));
        else            
            r1 = r1(~isnan(r1));
            r1(r1 == 1) = nan;
            % Consider thresholds in topCutoff vector.
            x = thresholdBetaScore(r1,[],[],topCutoff);
            % Correct using Bonferroni method.
            rho(rInd) = correctBetaPvalues( min(x), length(r1));
        end
    end
end