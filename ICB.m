function ICB(dataset)
% This code is the implementation of the paper : 
% Appearance-Based Person Re-identification by Intra-Camera Discriminative
% Models and Rank Aggregation, ICB (2015).
% If you find it useful, please cite our work. 

% @inproceedings{Prates:2015:ICB,
%   author = {R. F. de C. Prates and W. R. Schwartz},
%   booktitle = {International Conference on Biometrics},
%   title = {{Appearance-Based Person Re-identification by Intra-Camera Discriminative Models and Rank Aggregation}},
%   series = {Lecture Notes in Computer Science},
%   pages = {1-8},
%   year = {2015},
%   url = "papers/paper_2015_ICB_Prates.pdf",	
% }

% Any comments, sugestions and bugs reports : pratesufop@gmail.com
% Thanks! 

% default : viper and aggregation using Stuart's method
params.aggr_method = 'stuart';
params.dataset = dataset;

% Number of components in PLS.
params.comp=3;
params.compOSS=1;
params.epsilon = 10^-3;
params.maxstep =100;
% number of rounds + stability of the results.
params.nIter =10;
% Size of prototypes set.
params.k=25;

% The features employed in this work were obtained using the salient color
% names as described in the paper Salient Color Names for person
% re-identification, Yang, Y. et al. 2014, ECCV. 
% We reduced the number of components to 70 using PCA. 
%% Load partition
load(sprintf('auxiliary\\Partition_%s.mat',params.dataset));
params.saveDir = strcat('.\Graphics\ICB\',params.dataset,'\');
data_features = open(sprintf('./auxiliary\\%s_features_scncd.mat',params.dataset));
N = size(data_features.features,2)/2;
% loading the features
for i=1:N
    dataA(i,:) = [data_features.features(:,i)]';
    dataB(i,:) = [data_features.features(:,size(data_features.features,2)/2+i)]';
end
% store the cmc_curves
final_result= zeros(params.nIter, N/2); proposed_result= zeros(params.nIter, N/2);
probe_result= zeros(params.nIter, N/2); gallery_result = zeros(params.nIter, N/2);
kiss_result = zeros(params.nIter, N/2);

for iter =1 :params.nIter
    % Randomly splitting the dataset in training and testing data.
    params.idxtrain = trials(iter,1:N/2);
    params.idxtest = trials(iter,N/2 + 1:end);
    notsame = randperm(N/2,N/2);
    
    nDta = dataA(params.idxtrain,:);
    nDtb = dataB(params.idxtrain,:);
    testDta = dataA(params.idxtest,:);
    testDtb = dataB(params.idxtest,:);
    
    % Same individuals
    dataSame = bsxfun(@minus,nDta,nDtb); 
    % Normalization (zero mean)
    dataSame = bsxfun(@minus,dataSame,mean(dataSame));

    % Not same individuals
    dataNotSame = bsxfun(@minus,nDta,nDtb(notsame,:));
    % Normalization    
    dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
    
    % Using the kissme idea M = inv(same) - inv(not_same)
    KissML.model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
    [~,rank_kiss] = sort(sqdist(testDta',testDtb',KissML.model),2,'ascend');
    
    % Selecting the KNN
    testDtb = dataB(params.idxtest,:);
    neighbors = PLS2OSS(nDtb, testDtb, nDta, params);
    
    
    % PLS parameters for Prototypes Modeling
    PLSparams.factor = params.comp;
    PLSparams.epsilon = params.epsilon;
    PLSparams.maxstep = params.maxstep;
    
    modelGallery={}; 
    t_id = [1:numel(params.idxtrain)];
    for i=1:numel(params.idxtest)
         kNN = neighbors(i,:);
         %Training the models using prototypes
         neg_samples = nDta(ismember(t_id,kNN)~=1,:);
         pos_samples= nDta(kNN,:);

         nneg = size(neg_samples,1);
         npos=  size(pos_samples,1);

         data = [neg_samples; pos_samples];
         labels = [-ones(nneg,1); ones(npos,1)];

         modelGallery{i}= umd_pls_nipals2(data, labels, PLSparams);
    end
    
    % Doing the same for probe images
    neighbors = PLS2OSS(nDta, testDta, nDtb, params);
    modelProbe={};
    
    t_id = [1:numel(params.idxtrain)];
    for i=1:numel(params.idxtest)
         kNN = neighbors(i,:);
         %Training the models using prototypes
         neg_samples = nDtb(ismember(t_id,kNN)~=1,:);
         pos_samples= nDtb(kNN,:);

         nneg = size(neg_samples,1);
         npos=  size(pos_samples,1);

         data = [neg_samples; pos_samples];
         labels = [-ones(nneg,1); ones(npos,1)];

         modelProbe{i}= umd_pls_nipals2(data, labels, PLSparams);
    end
    
    % Result Curves
    result_probe = zeros(1,numel(params.idxtest));
    result_gallery = zeros(1,numel(params.idxtest));
    result_proposed = zeros(1,numel(params.idxtest));
    result_final = zeros(1,numel(params.idxtest));
    result_kiss = zeros(1,numel(params.idxtest));
    
    % Evaluate each probe image
    for p=1:numel(params.idxtest)

        % Projecting probe at each gallery Model
        projProbe2Gallery = []; projGallery2Probe =[];
        for n=1:numel(params.idxtest)
            Xnorm = bsxfun(@rdivide,bsxfun(@minus,testDta(p,:), modelGallery{n}.Xdata.mean),modelGallery{n}.Xdata.std);
            projProbe2Gallery(n) = (Xnorm * modelGallery{n}.Bstar .* modelGallery{n}.Ydata.std) +  modelGallery{n}.Ydata.mean;
        end
        
        for n=1:numel(params.idxtest)
            Xnorm = bsxfun(@rdivide,bsxfun(@minus,testDtb(n,:), modelProbe{p}.Xdata.mean),modelProbe{p}.Xdata.std);
            projGallery2Probe(n) = (Xnorm * modelProbe{p}.Bstar .* modelProbe{p}.Ydata.std) +  modelProbe{p}.Ydata.mean;
        end
        
        %% Aggregating the probe and gallery based ranking lists obtained
        [~,idx_P2G] = sort(projProbe2Gallery,'descend');
        [~,idx_G2P] = sort(projGallery2Probe,'descend');
        
        [~, pval, rowNames] = aggregateRanks({idx_P2G, idx_G2P}, numel(params.idxtest),params.aggr_method,1);
        [~,idx] = sort(pval);
        aggr_rank = rowNames(idx);
        
        %% Aggregating our proposed method with the method proposed in Yang et al.(2014)
        %% Salient color names for person re-identification ECCV - 2014
        nlist={};
        nlist{1}= rank_kiss(p,:);
        nlist{2}= aggr_rank;
        
        [~, pval, ~] = aggregateRanks(nlist, numel(params.idxtest),'stuart',1);
        [~,idx] = sort(pval);
        final_rank = idx;
        
        %% CMC curves
        result_kiss(rank_kiss(p,:)==p)= result_kiss(rank_kiss(p,:)==p) + 1; 
        result_probe(idx_P2G==p) = result_probe(idx_P2G==p) + 1; 
        result_gallery(idx_G2P==p) = result_gallery(idx_G2P==p) + 1; 
        result_proposed(aggr_rank==p) = result_proposed(aggr_rank==p) + 1;
        result_final(final_rank==p) = result_final(final_rank==p) + 1;
    end   
    
    gallery_result(iter,:) = cumsum(result_gallery)./numel(params.idxtest);
    probe_result(iter,:)= cumsum(result_probe)./numel(params.idxtest);
    proposed_result(iter,:)= cumsum(result_proposed)./numel(params.idxtest);
    final_result(iter,:)= cumsum(result_final)./numel(params.idxtest);
    kiss_result(iter,:) = cumsum(result_kiss)./numel(params.idxtest);
end

answer={}; legend = {};
answer{end+1} = mean(gallery_result,1);
legend{end+1} = 'Gallery';

answer{end+1} = mean(probe_result,1);
legend{end+1} = 'Probe';

answer{end+1} = mean(proposed_result,1);
legend{end+1} = 'Probe + Gallery';

answer{end+1} = mean(final_result,1);
legend{end+1} = 'Aggr.';

answer{end+1} = mean(kiss_result,1);
legend{end+1} = 'SCNCD';

% Ploting histograms, fusion and type-based features.
fileName = 'ICB Results';
ntitle = sprintf('CMC Curve - %s',upper(params.dataset));
PlotCurve(answer, params.saveDir, fileName, legend, ntitle); 
set(gcf,'color','w');
end

function  neighbors = PLS2OSS(trainDt, testDt, BackDt, params)

% PLS parameters for KNN PLS2OSS
PLSparams.factor = params.compOSS;
PLSparams.epsilon = params.epsilon;
PLSparams.maxstep = params.maxstep;

nG = size(testDt,1);
for j=1:nG
    data=[]; labels=[];
    % Preparing the data
    data = [testDt(j,:); BackDt];
    % Setting the labels for the OAA approach
    labels = [1; -ones(nG,1)]; 
    % Learning the model
    modelGallery{j}= umd_pls_nipals2(data, labels, PLSparams);
end

nT = size(trainDt,1);

for j=1: nT
    data =[]; labels=[];
    data = [trainDt(j,:); BackDt];
    labels = [1; -ones(nT,1)]; 
    modelTraining{j} = umd_pls_nipals2(data, labels, PLSparams);
end

resp = zeros(nG, nT);
for i=1:nG
    for j=1:nT
        %% Avaliating the training sample at the gallery model
        Xnorm = bsxfun(@rdivide,bsxfun(@minus,trainDt(j,:), modelGallery{i}.Xdata.mean),modelGallery{i}.Xdata.std);
        % project samples to the current model
        responseG  = (Xnorm * modelGallery{i}.Bstar .* modelGallery{i}.Ydata.std) +  modelGallery{i}.Ydata.mean;
        %% Avaliate the gallery sample at the training model
        Xnorm = bsxfun(@rdivide,bsxfun(@minus,testDt(i,:), modelTraining{j}.Xdata.mean),modelTraining{j}.Xdata.std);
        % project samples to the current model
        responseT = (Xnorm * modelTraining{j}.Bstar .* modelTraining{j}.Ydata.std) +  modelTraining{j}.Ydata.mean; 
        resp(i,j) = (responseT + responseG);
    end
end

% Storing only the K-NN
[~,idx] = sort(resp, 2, 'descend'); 

neighbors = idx(:,1:params.k);
end


function Model= umd_pls_nipals2(X,Y,params)

% --------------------- Initializations --------------------------
T=[];P=[];W=[];Wstar=[];U=[];b=[];C=[];R2_X=[];R2_Y=[];
% ----------------------------------------------------------------
nfactor = params.factor;
[X, M_X, S_X] = zscore(X);
[Y, M_Y, S_Y] = zscore(Y);

S_X(S_X < eps) = 1;
S_Y(S_Y < eps) = 1;

Xdata.mean = M_X; Xdata.std = S_X; Xdata.X =X;
Ydata.mean = M_Y; Ydata.std = S_Y; Ydata.Y = Y;

[nn,np]=size(X);
[n,nq]=size(Y);
if nn~= n;
    error(['Incompatible # of rows for X and Y']);
end
% Precision for convergence
epsilon=params.epsilon;

U=zeros(n,nfactor);
C=zeros(nq,nfactor);
% The X set
T=zeros(n,nfactor);
P=zeros(np,nfactor);
W=zeros(np,nfactor);
b=zeros(1,nfactor);


Xres=X;
Yres=Y;
for l=1:nfactor
    u=Yres(:,1); u =u./norm(u);
    u0=rand(n,1)*10; u0 = u0./norm(u0);
    t=u;
    nstep=0;
    maxstep=params.maxstep;
    while ( ( (u0-u)'*(u0-u) > epsilon/2) & (nstep < maxstep));
        nstep=nstep+1;
        disp(['Latent Variable #',int2str(l),'  Iteration #:',int2str(nstep)])
        u0=u;
        w=Xres'*u; 
        t=Xres*w; t = t./norm(t);
        c=Yres'*t; 
        u=Yres*c; u = u./norm(u);
    end;
   
    T(:,l)=t;
    U(:,l)=u;
    % deflation of X and Y
    Xres=Xres-t*t'*Xres;
    Yres=Yres-t*t'*Yres;
end

% Rosipal and Trejo
Wstar= X'*U;
Bstar = X'*U*inv(T'*X*X'*U)*T'*Y;   

Model.T = T;
Model.Wstar = Wstar;
Model.Bstar = Bstar;
Model.Xdata = Xdata;
Model.Ydata = Ydata;

end