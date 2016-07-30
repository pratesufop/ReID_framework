function ICPR2016(fileName,dataset)
% dataset viper or prid450s
% fileName is the name of the graphic archives, e.g.'myGraph'. In fact, it
% will generate one graphic for supervised and other to unsupervised
% % experiments (e.g., 'myGraph_supervised' and 'myGraph_unsupervised'). 

% This code implements the method described in our ICPR 2016 paper.
% Kernel Hierarchical PCA for Person Re-Identification.
% It is part of our Person Re-ID framework. Available at 
% https://github.com/pratesufop/ReID_framework

% Any questions, please contact me at pratesufop@gmail.com

% ------------------------------------ IMPORTANT ----------------

% Please cite our work if you use the following code or parts of it. Thanks!


% @inproceedings{Prates2016ICPR,
% title = {Kernel Hierarchical PCA for Person Re-Identification},
% author = {Raphael Prates and William Robson Schwartz},
% url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2016/07/kernelHPCA.pdf},
% year = {2016},
% date = {2016-12-13},
% booktitle = {23th International Conference on Pattern Recognition, ICPR 2016, Cancun, MEXICO, December 4-8, 2016.}
% }

% OBS 1: 

% We are using the features provided by the paper:
% ' An Enhanced Deep Feature Representation for Person Re-identification'
% Shangxuan Wu, Ying-Cong Chen, Xiang Li, Jin-Jie You, Wei-Shi Zheng
% WACV2016: IEEE Winter Conference on Applications of Computer Vision.

% If you use these features, please cite their work!



% -----------------------------------------------------------------

% cleaning 
close all
clc

if nargin <2
    error('You must specify the dataset and filename');
end
addpath(genpath('.\auxiliary'));

load(sprintf('./auxiliary/%s_lomo_mix_avss2016.mat',dataset));
params.saveDir = strcat('.\Graphics\',dataset,'\');
answer={};legend={};
params.Iter =10;

hpca = zeros(params.Iter,numel(trials(1).labelsAtest));
kernel_hpca = zeros(params.Iter,numel(trials(1).labelsAtest));

for iter=1:params.Iter
    
    params.idxtrain = trials(iter).labelsAtrain;
    params.idxtest = trials(iter).labelsAtest;
    feat.dataA =  Xview1(:,params.idxtrain)'; feat.dataB = Xview2(:,params.idxtrain)';
    test_feat.dataA = Xview1(:,params.idxtest)';test_feat.dataB = Xview2(:,params.idxtest)';
    params.N = numel(params.idxtrain) + numel(params.idxtest);

    data.featA = feat.dataA; data.featB = feat.dataB;
    data.test_featA = test_feat.dataA; data.test_featB = test_feat.dataB;
    
    disp('>Learning hierarchical PCA Model');
    % Paramters 
    PCaparams.epsilon= 10^-5; PCaparams.maxstep= 1000; 
    PCaparams.print =1; PCaparams.factor =100;
    
    % -------- Linear Model --------------
    hpca(iter,:) = run_hpca(data,PCaparams);
   
   disp('>Applying Kernel to Train and Test...');
   
   % Applying Kernel
   [train_a_ker, omegaA] = kernel_expchi2(feat.dataA,feat.dataA);
   [train_b_ker, omegaB] = kernel_expchi2(feat.dataB,feat.dataB);
   [test_a_ker] = kernel_expchi2(test_feat.dataA,feat.dataA,omegaA);
   [test_b_ker] = kernel_expchi2(test_feat.dataB,feat.dataB,omegaB);

    % Centralized Kernel
    kerneldata.norm_train_a_ker = centerK(train_a_ker);
    kerneldata.norm_train_b_ker = centerK(train_b_ker);
    kerneldata.norm_test_a_ker = centerK(test_a_ker);
    kerneldata.norm_test_b_ker = centerK(test_b_ker);

    disp('>Applying Kernel HPCA');
    % -------- Kernel Model --------------
    % Paramters 
    PCaparams.epsilon= 10^-5; PCaparams.maxstep= 1000; 
    PCaparams.print =1; PCaparams.factor =100; 
    
    kernel_hpca(iter,:)= run_kernel2hpca(kerneldata,PCaparams);
end

% Plotting Supervised
answer{end+1} = mean(hpca,1);legend{end+1} = 'HPCA';
answer{end+1} = mean(kernel_hpca,1);legend{end+1} = 'Kernel HPCA';

ntitle = sprintf('Cumulative Matching Characteristic (CMC) Curve - %s',dataset);
PlotCurve(answer, params.saveDir, [fileName '_icpr2016'], legend, ntitle); 
set(gcf,'color','w');

end

function cmc_iter = run_kernel2hpca(data, PCaparams)
    
    num_test = size(data.norm_train_a_ker,1);
    Model=  Kernel_HPCA(data.norm_train_a_ker,data.norm_train_b_ker,PCaparams);
    nXa = data.norm_test_a_ker*Model.Wx; nXb = data.norm_test_b_ker*Model.Wy;
    
    [~,idx] = sort(pdist2(nXa,nXb,'cosine'),2,'ascend');
   
    nresp = zeros(1,num_test);
    for i=1:num_test
        nresp(idx(i,:)==i) = nresp(idx(i,:)==i) + 1;
    end
    cmc_iter = cumsum(nresp)./num_test;
end


function cmc_iter = run_hpca(data, PCAparams)
    
    num_test = size(data.featA,1);
    
    Model = HPCA(data.featA,data.featB,PCAparams);
    
    txA = bsxfun(@rdivide,bsxfun(@minus, data.test_featA, Model.dataA.mean), Model.dataA.std);
    txB = bsxfun(@rdivide,bsxfun(@minus, data.test_featB,  Model.dataB.mean),Model.dataB.std);
    
    nXa = txA*Model.Wa; nXb = txB*Model.Wb;
    
    [~,idx] = sort(pdist2(nXa,nXb,'cosine'),2,'ascend');
   
    nresp = zeros(1,num_test);
    for i=1:num_test
        nresp(idx(i,:)==i) = nresp(idx(i,:)==i) + 1;
    end
    cmc_iter = cumsum(nresp)./num_test;

end

% Kernel function
function [D,md] = kernel_expchi2(X,Y,omega)
  D = zeros(size(X,1),size(Y,1));
  parfor i=1:size(Y,1)
    d = bsxfun(@minus, X, Y(i,:));
    s = bsxfun(@plus, X, Y(i,:));
    D(:,i) = sum(d.^2 ./ (s+eps), 2);
  end
  md = mean(mean(D));
  if nargin < 3
    omega = md;
  end
  D = exp( - 1/(2*omega) .* D);
end

function nK = centerK(K, m)
    if nargin < 2
        m = mean(K,1);
    end
    n = size(K,1);
    nK = bsxfun(@minus,K, m);
end



function Model= Kernel_HPCA(KXa,KXb,params)

nfactor = params.factor;
% Precision for convergence
epsilon=params.epsilon;

n = size(KXa,1);
Xa_res = KXa;  Xb_res = KXb; 
for l=1:nfactor
    super_t=Xa_res(:,1);
    super_t0=rand(n,1)*10; super_t0 = super_t0./norm(super_t0);
    nstep=0;
    maxstep=params.maxstep;
    while ( ( (super_t0-super_t)'*(super_t0-super_t) > epsilon/2) & (nstep < maxstep));
        nstep=nstep+1;
        disp(['Latent Variable #',int2str(l),'  Iteration #:',int2str(nstep)])
        super_t0=super_t;
        %computing the loadings
        ta = Xa_res*super_t;  ta = ta./norm(ta);
        tb = Xb_res*super_t; tb = tb./norm(tb);
        T = [ta, tb];
        wt = T'*super_t; 
        super_t = T*wt; super_t = super_t./norm(super_t);
    end
    sT(:,l) = super_t;
    % deflation of X and Y
    Xa_res=Xa_res-super_t*super_t'*Xa_res;
    Xb_res=Xb_res-super_t*super_t'*Xb_res;
end

Model.Wx = sT*inv(sT'*KXa*sT);
Model.Wy = sT*inv(sT'*KXb*sT);
end

function Model= HPCA(Xa,Xb,params)

[Xa, M_Xa, S_Xa] = zscore(Xa);
[Xb, M_Xb, S_Xb] = zscore(Xb);

nfactor = params.factor;

S_Xa(S_Xa < eps) = 1;
S_Xb(S_Xb < eps) = 1;


Model.dataA.mean = M_Xa; Model.dataA.std = S_Xa; 
Model.dataB.mean = M_Xb; Model.dataB.std = S_Xb; 

% Precision for convergence
epsilon=params.epsilon;

n = size(Xa,1);
Xa_res = Xa; 
Xb_res = Xb; 
for l=1:nfactor
    super_t=Xa_res(:,1);
    super_t0=rand(n,1)*10; super_t0 = super_t0./norm(super_t0);
    nstep=0;
    maxstep=params.maxstep;
    while ( ( (super_t0-super_t)'*(super_t0-super_t) > epsilon/2) & (nstep < maxstep));
        nstep=nstep+1;
        disp(['Latent Variable #',int2str(l),'  Iteration #:',int2str(nstep)])
        super_t0=super_t;
        %computing the loadings
        wa = Xa_res'*super_t; 
        wb = Xb_res'*super_t;
        ta = Xa_res*wa; ta = ta./norm(ta);
        tb = Xb_res*wb; tb = tb./norm(tb);
        T = [ta, tb];
        wt = T'*super_t;
        super_t = T*wt; super_t = super_t./norm(super_t);
    end
    % deflation of X and Y
    Xa_res=Xa_res-super_t*super_t'*Xa_res;
    Xb_res=Xb_res-super_t*super_t'*Xb_res;
    sT(:,l) = super_t;
end
Model.Wa = Xa'*sT*inv(sT'*Xa*Xa'*sT);  
Model.Wb = Xb'*sT*inv(sT'*Xb*Xb'*sT);
end