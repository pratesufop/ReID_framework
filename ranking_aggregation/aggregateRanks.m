function [aggR, pval, rowNames] = aggregateRanks(R, N, method, complete, topCutoff)
%AGGREGATERANKS Aggregate ranked lists using traditional and robust methods
%   [aggR, pval, rowNames] = AGGREGATERANKS(R,N,method,complete,topCutoff)
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   INPUTS:
%       R           -> matrix representation: numeric matrix with as many
%                   rows as the number of unique elements and with as many
%                   columns as the number of ranked lists. All entries have
%                   to be on interval [0,1]. Smaller numbers correspond to
%                   better ranking.
%                   -> list representation: cell array of
%                   cells with strings or vectors with numbers - in this
%                   case R is transformed into numeric rank matrix.
%       N           number of ranked elements, default is the number of
%                   unique elements in R
%       method      rank aggregation method. Could be one of the following:
%                   'min', 'median', 'mean', 'geom.mean', 'stuart', 'RRA'.
%                   Default is 'RRA' (Robust Rank Aggregation).
%       complete    1 - rankings are complete (there is perfect match
%                   between sets of rankings)
%                   0 - default; rankings are incomplete.
%       topCutoff   vector of cutoff values that limit the number of
%                   elements in the input list.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   OUTPUTS:
%       aggR        vector of aggregated ranks/scores (it equals pval for
%                   methods 'stuart' and'RRA')
%       pval        p-values (relevant only for 'mean','stuart','RRA')
%       rowNames    if R contains lists, rowNames contains their unique
%                   names in the same order as the values of aggR
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   EXAMPLES
%       % Lets have three ordered lists of names.
%       R = {   {'Joe', 'Bob', 'Lucy', 'Mary'}, ...
%               {'Bob', 'Joe', 'Lucy', 'Mary'}, ...
%               {'Lucy', 'Mary', 'Bob', 'Joe'}}
%
%       % We can also use numerical vectors instead of cell of strings.
%       % R = { [1,2,3,4], [2,1,3,4], [3,4,2,1] }
%
%       % Obtain aggregated ranks with method 'RRA' (default).
%       [aggR, pval, rowNames] = aggregateRanks(R)
%
%       % Or, equivalently, use explicit parameters definition.
%       [aggR, pval, rowNames] = aggregateRanks(R, [], 'RRA')
%
%       % We can also compute a matrix with ranks first ...
%       rmat = rankMatrix(R)
%       % ... and then pass it to the aggregation method.
%       [aggR, pval, rowNames] = aggregateRanks(rmat)
%
%       % A case of incomplete lists.
%       R = {   {'Joe', 'Bob', 'Lucy', 'Mary'}, ...
%               {'Bob', 'Joe', 'Lucy',       }, ...
%               {'Lucy', 'Mary'              }}
%
%       % Lets compute mean ranks. Mind the fourth parameter, which
%       % indicates completeness of the lists. Note the return values; aggR
%       % contains average across the ranks, while pval contains the
%       % statistical significance (p-value) of mean ranks.
%       [aggR, pval, rowNames] = aggregateRanks(R,[],'mean',0)
%
%       % We can also say that only top k elements are presented in data
%       % by setting the topCutoff to [1,0.75,0.5].
%       [aggR, pval, rowNames] = aggregateRanks(R,[],'RRA',0,[1,0.75,0.5])
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   See also RANKMATRIX.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   Copyright (2013) Nejc Ilc <nejc.ilc@gmail.com>
%   Based on R package RobustRankAggreg written by Raivo Kolde.
%   Reference:
%     Kolde, R., Laur, S., Adler, P., & Vilo, J. (2012).
%     Robust rank aggregation for gene list integration and meta-analysis.
%     Bioinformatics, 28(4), 573-580
%   Revision: 1.0 Date: 2013/05/16
%--------------------------------------------------------------------------

rowNames = [];

if ~exist('R','var') || isempty(R)
    error('Input parameter R is missing!');
end

if ~exist('N','var')
    N=[];
end

if ~exist('complete','var') || isempty(complete)
    complete = 0;
end

% Input parameter R determination
if iscell(R)
    [rmat, rowNames] = rankMatrix(R, N, complete);
elseif ismatrix(R)
    if all(max(R,[],1)<=1) && all(min(R,[],1)>0)
        rmat = R;
    else
        error('Columns of matrix R can only contain numbers from interval (0,1].');
    end
else
    error('R should be cell (of lists) or matrix (of ranks).');
end

if ~exist('method','var') || isempty(method)
    method = 'RRA';
end

if ~exist('topCutoff','var') || isempty(topCutoff)
    topCutoff = NaN;
end

pval = NaN;

switch method
    
    case 'min'
        aggR = min(rmat,[],2);
        
    case 'median'
        aggR = nanmedian(rmat,2);
        
    case 'geom.mean'
        aggR = exp(nanmean(log(rmat),2));
        
    case 'mean'
        aggR = nanmean(rmat, 2);
        n = sum(~isnan(rmat),2);
        pval = normcdf(aggR, 0.5, sqrt(1/12./n));
        
    case 'stuart'
        aggR = stuart(rmat);
        pval = aggR;
        
    case 'RRA'
        aggR = rhoScores(rmat, topCutoff);
        pval = aggR;
        
    otherwise
        error('Method should be one of:  "min", "geom.mean", "mean", "median", "stuart" or "RRA"');
end