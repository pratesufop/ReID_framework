function [rmat, rowNames] = rankMatrix(glist, N, complete)
%RANKMATRIX Transform ranked lists into rank matrix
%   [rmat, rowNames] = RANKMATRIX(glist, N, complete) 
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -   
%   INPUTS:
%       glist       cell array of cells with strings or numerical vectors.
%       N           number of ranked elements, default is the number of
%                   unique elements in R
%       complete    1 - rankings are complete (there is perfect match
%                   between sets of rankings) 
%                   0 - default; rankings are incomplete.
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   OUTPUTS:
%       rmat        numeric matrix with as many rows as the number of
%                   unique elements and with as many columns as the number 
%                   of ranked lists. All entries have to be on interval 
%                   [0,1]. Smaller numbers correspond to better ranking.
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
%       % Compute a matrix with ranks.
%       rmat = rankMatrix(R)
%
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%   See also AGGREGATERANKS.
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
    
    if ~exist('complete','var') || isempty(complete)
        complete = 0;
    end
    
    nCell = length(glist);
    NperCell = cellfun(@length, glist);
    
    % convert each cell in glist to row vector
    glist = cellfun(@(c) c(:)', glist, 'UniformOutput',false);
    glist_all = [glist{:}];
  
    % find unique elements and map strings to numbers (ib_map)
    [u_map, ~, ib_map]= unique(glist_all);
    realLen = length(u_map);
    
    if ~exist('N','var') || isempty(N)
        N = realLen;
    end
    
    % complete each list with NaNs to match its length with realLen
    M = nan(realLen, nCell);
    start = 1;
    for c=1:nCell
        stop = start + NperCell(c)-1;
        M(1:NperCell(c),c) = ib_map(start:stop);
        start = stop + 1;
    end

    [U,~,ib] = unique(M);
    fullLen = length(U);
    
    if ~complete
        rmat = ones(fullLen, nCell)*N;
        N = ones(1,nCell)*N;
    else
        rmat = nan(fullLen, nCell);
        N = NperCell;
    end

    Umat = reshape(ib,realLen,nCell);
    v = (1:realLen)';
    
    for i=1:nCell
        u=Umat(:,i);
        rmat(u,i) = v;
    end
    
    rmat = rmat(1:realLen,:);
    rmat = bsxfun(@rdivide,rmat,N);

    rowNames = u_map(1:realLen)';
end