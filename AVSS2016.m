function AVSS2016(fileName,dataset)
% dataset viper or prid450s
% fileName is the name of the graphic archives, e.g.'myGraph'. In fact, it
% will generate one graphic for supervised and other to unsupervised
% experiments (e.g., 'myGraph_supervised' and 'myGraph_unsupervised'). 

% This code implements the method described in our AVSS 2016 paper.
% Kernel Partial Least Squares for Person Re-Identification.
% It is part of our Person Re-ID framework. Available at 
% https://github.com/pratesufop/ReID_framework


% Any questions, please contact me at pratesufop@gmail.com

% ------------------------------------ IMPORTANT ----------------

% Please cite our work if you use the following code or parts of it. Thanks!

% @inproceedings{Prates2016AVSS,
% title = {Kernel Partial Least Squares for Person Re-Identification},
% author = {Raphael Prates and Marina Oliveira and William Robson Schwartz},
% url = {http://www.ssig.dcc.ufmg.br/wp-content/uploads/2016/07/egpaper_for_DoubleBlindReview.pdf},
% year = {2016},
% date = {2016-08-24},
% booktitle = {IEEE International Conference on Advanced Video and Signal-Based Surveillance (AVSS)}
% }


% OBS 1: 
% In the paper, we used a different feature for VIPER dataset. Recently, we
% performed new experiments using a combination of CNN featrues and LOMO
% (presented by the paper "An enhanced deep feature representation for
% person re-identification") obtaining improved results. 
% Therefore, this code obtain results improved when compared to the paper,
% but only due to the feature descriptor.

% OBS 2:
% It can be small difference (~1%) in the values reported in the paper and those
% presented here due to different partitions. Here, we fixed the
% partitions, but before it was randomically generated without saving.

% OBS 3:
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

params.ModeAfactors = 125; params.XKPLSfactors = 100;

kplsMode = zeros(params.Iter,numel(trials(1).labelsAtest));
xkpls = zeros(params.Iter,numel(trials(1).labelsAtest)); 
plsOAA = zeros(params.Iter,numel(trials(1).labelsAtest));
kpls = zeros(params.Iter,numel(trials(1).labelsAtest));

for iter=1:params.Iter
    
    params.idxtrain = trials(iter).labelsAtrain;
    params.idxtest = trials(iter).labelsAtest;
    feat.dataA =  Xview1(:,params.idxtrain)'; feat.dataB = Xview2(:,params.idxtrain)';
    test_feat.dataA = Xview1(:,params.idxtest)';test_feat.dataB = Xview2(:,params.idxtest)';
    params.N = numel(params.idxtrain) + numel(params.idxtest);

   disp('>Applying Kernel to Train and Test...');
   
   % Applying Kernel
   [train_a_ker, omegaA] = kernel_expchi2(feat.dataA,feat.dataA);
   [train_b_ker, omegaB] = kernel_expchi2(feat.dataB,feat.dataB);
   [test_a_ker] = kernel_expchi2(test_feat.dataA,feat.dataA,omegaA);
   [test_b_ker] = kernel_expchi2(test_feat.dataB,feat.dataB,omegaB);

    % Centralized Kernel
    mean_a = mean(train_a_ker,1); mean_b = mean(train_b_ker,1); 
    kerneldata.norm_train_a_ker = centerK(train_a_ker,mean_a);
    kerneldata.norm_train_b_ker = centerK(train_b_ker,mean_b);
    kerneldata.norm_test_a_ker = centerK(test_a_ker, mean_a);
    kerneldata.norm_test_b_ker = centerK(test_b_ker, mean_b);

    disp('>Applying KPLS Mode A');
    % Parameters of KPLS Mode A model
    KPLSModeA_params.epsilon= 10^-6; KPLSModeA_params.maxstep= 1000; 
    KPLSModeA_params.print =1; KPLSModeA_params.nfactor = params.ModeAfactors;
      
    kplsMode(iter,:) =  run_kpls2modeA(kerneldata,KPLSModeA_params);
    
    disp('>Applying KPLSD');
    % Paramters 
    XKPLSparams.epsilon= 10^-6; XKPLSparams.maxstep= 1000; 
    XKPLSparams.print =1; XKPLSparams.factor = params.XKPLSfactors;
    
    xkpls(iter,:) = run_XKPLS(kerneldata,XKPLSparams);  
    
    % -------------------- UNSUPERVISED EXPERIMENTS ---------
    disp('>Applying PLS OAA at gallery images');
    % Paramters 
    PLSparams.epsilon= 10^-6; PLSparams.maxstep= 1000; 
    PLSparams.print =1; PLSparams.factor = 10;

    plsOAA(iter,:) = run_plsOAA(test_feat.dataA,test_feat.dataB,PLSparams);
    
    disp('>Applying KPLS');
    % Paramters 
    KPLSparams.epsilon= 10^-6; KPLSparams.maxstep= 1000; KPLSparams.print =1; 
    if strcmp(dataset,'viper')
        KPLSparams.factor = 300;
    else
        KPLSparams.factor = 200;
    end
    
%     Observe that the kernel computation is different here!
    [test_a_ker, ~] = kernel_expchi2(test_feat.dataA, test_feat.dataB);
    [test_b_ker, mean_b] = kernel_expchi2(test_feat.dataB, test_feat.dataB);
        
    test_b_ker = centerK(test_b_ker, mean_b);
    test_a_ker = centerK(test_a_ker, mean_b);

    %% Equation (11) 
%     n = size(test_a_ker,1); nt = size(test_a_ker,1);
%     test_a_ker = (test_a_ker -(1/n).*ones(nt,1)*ones(n,1)'*test_b_ker)*(eye(n,n) - (1/n).*ones(n,1)*ones(n,1)');
%     
    kpls(iter,:) = run_kpls(test_a_ker,test_b_ker,KPLSparams);
end

% Plotting Supervised
answer{end+1} = mean(kplsMode,1);legend{end+1} = 'KPLS-Mode A';
answer{end+1} = mean(xkpls,1);legend{end+1} = 'X-KPLS';

ntitle = sprintf('Cumulative Matching Characteristic (CMC) Curve - %s',dataset);
PlotCurve(answer, params.saveDir, [fileName '_supervised'], legend, ntitle); 
set(gcf,'color','w');

% Plotting Unsupervised
answer={}; legend={};
answer{end+1} = mean(plsOAA,1);legend{end+1} = 'PLS OAA';
answer{end+1} = mean(kpls,1);legend{end+1} = 'KPLS';

ntitle = sprintf('Cumulative Matching Characteristic (CMC) Curve - %s',dataset);
PlotCurve(answer, params.saveDir, [fileName '_unsupervised'] , legend, ntitle); 
set(gcf,'color','w');

end

function cmc_iter = run_kpls(norm_test_a_ker,norm_test_b_ker, PLSparams)
    
    num_test = size(norm_test_b_ker,1);
    
    Y = eye(num_test);
    Model= KPLS_NIPALS(norm_test_b_ker,Y,PLSparams);
    
    % Projecting probe and gallery in the learned regression model
    Yhat = norm_test_b_ker*Model.W; Yhat = normr(Yhat);
    Ystar = norm_test_a_ker*Model.W; Ystar = normr(Ystar);
   
    resp = pdist2(Ystar,Yhat,'cosine'); [~,idx]=sort(resp,2,'ascend');
    nresp = zeros(1,num_test);
    for i=1:num_test
        nresp(idx(i,:)==i) = nresp(idx(i,:)==i) + 1;
    end
    cmc_iter = cumsum(nresp)./num_test;
end 

function cmc_iter = run_plsOAA(test_featA, test_featB, PLSparams)
    
    num_test = size(test_featB,1);
    
    dist_probe2gallery = zeros(num_test);
    parfor i=1:num_test
        Y = -ones(1,num_test); Y(i)=1;
        % Learning the model
        Model = umd_pls_nipals2(test_featB,Y',PLSparams);
        % Computing the novel gallery representation
        gallery =  ((test_featB(i,:) - Model.Xdata.mean)./Model.Xdata.std)*Model.Wstar;
        % Projecting all probes in the learned Model.
        probe = bsxfun(@rdivide,bsxfun(@minus,test_featA,Model.Xdata.mean),Model.Xdata.std)*Model.Wstar;
        dist_probe2gallery(:,i) = pdist2(gallery, probe);
    end
    nresp = zeros(1,num_test);
    [~, idx] = sort(dist_probe2gallery,2,'ascend');
   
    for i=1: num_test
        nresp(idx(i,:)==i) = nresp(idx(i,:)==i) + 1;
    end
    cmc_iter = cumsum(nresp)./num_test;
end

function cmc_iter = run_kpls2modeA(data,PLSparams)
    %  NIPALS KPLS mode A
    tic
    KPLS_ModelA = NIPALS_KPLS_ModeA(data.norm_train_a_ker,data.norm_train_b_ker,PLSparams);
    toc
    % Projecting test samples
    latentA = data.norm_test_a_ker*KPLS_ModelA.P/(KPLS_ModelA.P'*KPLS_ModelA.P);
    latentB = data.norm_test_b_ker*KPLS_ModelA.Q/(KPLS_ModelA.Q'*KPLS_ModelA.Q);

    num_test = size(data.norm_test_a_ker,1);
    [~,rank] = sort(pdist2(latentA,latentB,'cosine'),2,'ascend');
    responsePLS = zeros(1,num_test);
    for i=1:num_test
        responsePLS(rank(i,:)==i) = responsePLS(rank(i,:)==i) + 1;
    end
    
    cmc_iter = cumsum(responsePLS)./num_test;
end

function cmc_iter = run_XKPLS(data,PLSparams )
    % Learning the X-KPLS Model
    Y = eye(size(data.norm_train_b_ker));
    ModelB= KPLS_NIPALS(data.norm_train_b_ker,Y,PLSparams);

     Y = eye(size(data.norm_train_a_ker));
    ModelA= KPLS_NIPALS(data.norm_train_a_ker,Y,PLSparams);
    toc

    num_test = size(data.norm_test_a_ker,1);
    
    YstarA = data.norm_test_a_ker*ModelA.W; YstarA = normr(YstarA);
    YstarB = data.norm_test_b_ker*ModelB.W; YstarB = normr(YstarB);

    resp = pdist2(YstarA,YstarB,'cosine'); [~,idx]=sort(resp,2,'ascend');
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




function Model= KPLS_NIPALS(XK,Y,PLSparams)
% Raphael Prates
% KPLS model test

Ydata.data = Y;
%% centralize the data
[nn,~]=size(XK);
[n,~]=size(Y);
if nn~= n;
    error(['Incompatible # of rows for X and Y']);
end
% Precision for convergence
epsilon=PLSparams.epsilon;
nfactor = PLSparams.factor;
% The Y set
U=zeros(n,nfactor);
% The X set
T=zeros(n,nfactor);
Kc=XK;
Yres=Y;

for iter=1:nfactor
    %% 1. Randomly initialize u 
    u=Yres(:,1);
    u0=rand(n,1)*10; u0 =u0./norm(u0);
    nstep=0;
    maxstep=PLSparams.maxstep;
    while ( ( (u0-u)'*(u0-u) > epsilon) && (nstep < maxstep))
        nstep=nstep+1;
        disp(['Latent Variable #',int2str(iter),'  Iteration #:',int2str(nstep)])
        u0=u;
        t=Kc*u; t =t./norm(t);
        c = Yres'*t; 
        u=Yres*c; u =u./norm(u);
    end
    % Store in matrices
    T(:,iter)=t;U(:,iter)=u;
    % deflation of X and Y
	Kc=Kc-t*t'*Kc - Kc*t*t' + t*t'*Kc*t*t';
    Yres=Yres- t*t'*Yres;
end
Model.W = U*inv(T'*XK*U)*T'*Y;
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

Xdata.mean = M_X; Xdata.std = S_X;
Ydata.mean = M_Y; Ydata.std = S_Y;

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
    t=Yres(:,1); t =t./norm(t);
    t0=rand(n,1)*10; t0 = t0./norm(t0);
    u=t;
    nstep=0;
    maxstep=params.maxstep;
    while ( ( (t0-t)'*(t0-t) > epsilon/2) & (nstep < maxstep));
        nstep=nstep+1;
        disp(['Latent Variable #',int2str(l),'  Iteration #:',int2str(nstep)])
        t0=t;
        w=Xres'*u; w = w./norm(w);
        t=Xres*w; t = t./norm(t);
        c=Yres'*t; c = c./norm(c);
        u=Yres*c;
    end;
    p=Xres'*t;
    % b coef
    b_l=((t'*t)^(-1))*(u'*t);
    b_1=u'*t;
    % Store in matrices
    b(l)=b_l;
    P(:,l)=p;
    W(:,l)=w;
    T(:,l)=t;
    U(:,l)=u;
    C(:,l)=c;
    % deflation of X and Y
    Xres=Xres-t*p';
    Yres=Yres-(b(l)*(t*c'));
end

% The Wstart weights gives T=X*Wstar
Wstar=W*inv(P'*W);
Bstar = Wstar*inv(T'*T)*T'*Y; 

Model.T = T;
Model.Wstar = Wstar;
Model.Bstar = Bstar;
Model.Xdata = Xdata;
Model.Ydata = Ydata;

end


function Model = NIPALS_KPLS_ModeA(Kx,Ky,params)
% Raphael Prates (pratesufop@gmail.com) 
% KPLS mode A NIPALS
% 1) randomly initialize u;
% 2) It computes t = XX'u = (Kx*u) and normalize t.
% 3) It computes u = YY't = (Ky*t) and normalize u.
% 4) Deflates Kx and Ky 
% (Kx = Kx - t*inv(t't)t'Kx) and (Ky = Ky - u*inv(u'u)u'Ky)
% --------------------- Initializations --------------------------
T=[];P=[];U=[];Q=[];
% ----------------------------------------------------------------

[nn,np]=size(Kx);
[n,nq]=size(Ky);
if nn~= n;
    error(['Incompatible # of rows for X and Y']);
end

U=zeros(n,params.nfactor);
T=zeros(n,params.nfactor);
P=zeros(np,params.nfactor);
Q=zeros(nq,params.nfactor);

Kxres=Kx;
Kyres=Ky;
for iter=1:params.nfactor
    t=Kyres(:,1); t =t./norm(t);
    t0=rand(n,1)*10; t0 = t0./norm(t0);
    u=t;
    nstep=0;
    while ( ( (t0-t)'*(t0-t) > params.epsilon) && (nstep < params.maxstep))        
        nstep=nstep+1;
        if params.print
            (t0-t)'*(t0-t)
            disp(['Latent Variable #',int2str(iter),'  Iteration #:',int2str(nstep)])
        end
        t0=t;
        t = Kxres*u; t = t./norm(t);
        u = Kyres*t; u = u./norm(u);
    end
    T(:,iter)=t;U(:,iter)=u;
	Kxres=Kxres-t*inv(t'*t)*t'*Kxres;
    Kyres=Kyres-u*inv(u'*u)*u'*Kyres;
end
P = Kx'*T/(T'*T);
Q = Ky'*U/(U'*U);

Model.T= T;Model.U= U;
Model.P= P; Model.Q= Q;
end
