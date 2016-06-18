function ICIP(dataset)

% ICIP 2015 Code
% Loading the color names and the foreground masks
load('auxiliary\trCoe.mat'); 
% Parameters
%% Define the parameters.
% Dataset (VIPer or PRID450S)
params.dataset = dataset;
params.colorsTable = w2c;
params.nIter = 10;
% image size and the number of parts.
params.image = [128,48];
params.nparts = 6;
params.hist_bins = 32;

params.hist ={'RGB','nrgb','HSV','L1L2L3'};
% params.hist ={'RGB','HSV'};
params.cn ={'RGB','nrgb','HSV','L1L2L3'};
% params.cn ={'RGB','HSV'};
% Based on the paper 
params.Isolated_Factors = 34;
params.Fusion = 70;
params.Type_Factors = 50;

params.aggr_method = 'stuart';

%% Load partition
load(sprintf('auxiliary\\Partition_%s.mat',params.dataset));
params.saveDir = strcat('.\Graphics\ICIP\',params.dataset,'\');

% Load the images and Masks(VIPER and PRID450S datasets). 
% N corresponds to the number of individuals 
[N, imgA, imgB] = read(params.image,params.dataset);
% Foreground Mask
load(sprintf('./Masks/%s.mat',params.dataset));
[maskA ,maskB] = splitMasks(msk);
% Gaussian Mask
params.Gauss_mask = getGauss(params.image);

% Extract Color Names Features.
scncd_nfeat_img = getSCNCDImg(params, imgA, imgB, params.Gauss_mask);
scncd_nfeat_fore = getSCNCDFore(params, imgA, imgB, maskA, maskB);

% Extract Histograms features.
hist_nfeat_img = getHistImg(params, imgA, imgB, params.Gauss_mask);
hist_nfeat_fore = getHistFore(params, imgA, imgB, maskA, maskB);

% Response for all curves.
resp_type_hist = zeros(params.nIter,N/2);
resp_type_scncd = zeros(params.nIter,N/2);
response_hist = zeros(numel(params.hist),params.nIter,N/2);
response_scncd = zeros(numel(params.cn),params.nIter,N/2);
response_fusion=zeros(params.nIter,N/2);
response_aggr_cn = zeros(params.nIter,N/2);
response_aggr_hist = zeros(params.nIter,N/2);
response_final = zeros(params.nIter,N/2);

for iter=1:params.nIter
    %% Learn the metric distance (KISSME).
    % Histograms (Isolated)
    params.idxtrain = trials(iter,1:N/2);
    params.idxtest = trials(iter,N/2 + 1:end);
    notsame = randperm(N/2,N/2);
    mHist={}; mSCNCD={};
 
    % # HISTOGRAMS #   
    % Feature Type-Based Combination
    hist_typeBasedA.train =[]; hist_typeBasedB.train =[];
    hist_typeBasedA.test =[]; hist_typeBasedB.test =[];
    
    for h=1:numel(params.hist)
        disp('Training a novel Model...');
        M=[];
        % Applying PCA (KISSME only works for low-dimensional data)
        coeff = pca([squeeze(hist_nfeat_img.dataA(params.idxtrain,h,:)),squeeze(hist_nfeat_fore.dataA(params.idxtrain,h,:));...
                     squeeze(hist_nfeat_img.dataB(params.idxtrain,h,:)),squeeze(hist_nfeat_fore.dataB(params.idxtrain,h,:))]);
        % Projecting data at a low-dimensional space
        coeff = coeff(:,1:params.Isolated_Factors);
        nDtA = [squeeze(hist_nfeat_img.dataA(params.idxtrain,h,:)),squeeze(hist_nfeat_fore.dataA(params.idxtrain,h,:))];
        hist_typeBasedA.train = [nDtA,hist_typeBasedA.train];
        
        % normalizing training data at camera A and projecting at the
        % low-dimensional space.
        meanA = mean(nDtA); 
        nDtA = bsxfun(@minus,nDtA,meanA); nDtA = nDtA*coeff;
        
        nDtB = [squeeze(hist_nfeat_img.dataB(params.idxtrain,h,:)),squeeze(hist_nfeat_fore.dataB(params.idxtrain,h,:))];
        hist_typeBasedB.train = [nDtB,hist_typeBasedB.train];
        
        % normalizing data at camera B.
        meanB = mean(nDtB);
        nDtB = bsxfun(@minus,nDtB,meanB); nDtB = nDtB*coeff;
        
        % Same individuals
        dataSame = bsxfun(@minus,nDtA,nDtB); 
        % Normalization (zero mean)
        dataSame = bsxfun(@minus,dataSame,mean(dataSame));
        
        % Not same individuals
        dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
        % Normalization    
        dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
       
        % Model Information
        testDtA = [squeeze(hist_nfeat_img.dataA(params.idxtest,h,:)),squeeze(hist_nfeat_fore.dataA(params.idxtest,h,:))]; 
        % Concatenation of histograms in cam A - hist_typeBasedA =[ hist1, hist2, ..., histm]
        hist_typeBasedA.test = [testDtA,hist_typeBasedA.test];        
        testDtA = bsxfun(@minus,testDtA,meanA); testDtA = testDtA*coeff;
        
        testDtB = [squeeze(hist_nfeat_img.dataB(params.idxtest,h,:)),squeeze(hist_nfeat_fore.dataB(params.idxtest,h,:))]; 
        % Concatenation of histograms in cam B - hist_typeBasedB =[ hist1, hist2, ..., histm]
        hist_typeBasedB.test = [testDtB,hist_typeBasedB.test];        
        testDtB = bsxfun(@minus,testDtB,meanB); testDtB = testDtB*coeff;
        
        M.meanA = meanA; M.meanB = meanB;
        M.coeff = coeff;
        M.testA = testDtA; M.testB = testDtB;
        % Using the kissme idea M = inv(same) - inv(not_same)
        M.model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
        M.name = sprintf('HIST. %s',params.hist{h});
        mHist{h} = M;
    end
    
    % Type-Based Model Evaluation
    coeff = pca([hist_typeBasedA.train;hist_typeBasedB.train]);
    % Projecting data at a low-dimensional space
    coeff = coeff(:,1:params.Type_Factors);
    meanA = mean(hist_typeBasedA.train); 
    nDtA = bsxfun(@minus,hist_typeBasedA.train,meanA); nDtA = nDtA*coeff;
        
    meanB = mean(hist_typeBasedB.train); 
    nDtB = bsxfun(@minus,hist_typeBasedB.train,meanB); nDtB = nDtB*coeff;
    
    % Same individuals
    dataSame = bsxfun(@minus,nDtA,nDtB); 
    % Normalization (zero mean)
    dataSame = bsxfun(@minus,dataSame,mean(dataSame));

    % Not same individuals
    dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
    % Normalization    
    dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
    
    testDtA = bsxfun(@minus,hist_typeBasedA.test,meanA); testDtA = testDtA*coeff;
    testDtB = bsxfun(@minus,hist_typeBasedB.test,meanB); testDtB = testDtB*coeff;
    
    typeRhist=[];
    model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
    [~,typeRhist] = sort(sqdist(testDtA',testDtB',model),2,'ascend');
    
    for i=1:size(typeRhist,2)
        resp_type_hist(iter, typeRhist(i,:)==i) = resp_type_hist(iter,typeRhist(i,:)==i) + 1; 
    end
    
    % # COLOR NAMES #
    
    % Feature Type-Based Combination
    cn_typeBasedA.train =[]; cn_typeBasedB.train =[];
    cn_typeBasedA.test =[]; cn_typeBasedB.test =[];
    
    %% Color Names (Isolated) Foreground
    for h=1:numel(params.cn)
        M=[];
        % Applying PCA (KISSME only works for low-dimensional data)
        coeff = pca([squeeze(scncd_nfeat_img.dataA(params.idxtrain,h,:)),squeeze(scncd_nfeat_fore.dataA(params.idxtrain,h,:));...
                     squeeze(scncd_nfeat_img.dataB(params.idxtrain,h,:)),squeeze(scncd_nfeat_fore.dataB(params.idxtrain,h,:))]);
        % Projecting data at a low-dimensional space
        coeff = coeff(:,1:params.Isolated_Factors);
        
        nDtA = [squeeze(scncd_nfeat_img.dataA(params.idxtrain,h,:)),squeeze(scncd_nfeat_fore.dataA(params.idxtrain,h,:))];
        cn_typeBasedA.train = [nDtA,cn_typeBasedA.train];
        
        meanA = mean(nDtA);
        nDtA = bsxfun(@minus,nDtA,meanA); nDtA = nDtA*coeff;
        
        nDtB = [squeeze(scncd_nfeat_img.dataB(params.idxtrain,h,:)),squeeze(scncd_nfeat_fore.dataB(params.idxtrain,h,:))];
        cn_typeBasedB.train = [nDtB, cn_typeBasedB.train];
        
        meanB = mean(nDtB); 
        nDtB = bsxfun(@minus,nDtB,meanB); nDtB = nDtB*coeff;
        
        % Same individuals
        dataSame = bsxfun(@minus,nDtA,nDtB); 
        % Normalization (zero mean)
        dataSame = bsxfun(@minus,dataSame,mean(dataSame));
        
        % Not same individuals
        dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
        % Normalization    
        dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
        
        testDtA = [squeeze(scncd_nfeat_img.dataA(params.idxtest,h,:)),squeeze(scncd_nfeat_fore.dataA(params.idxtest,h,:))]; 
        cn_typeBasedA.test = [testDtA, cn_typeBasedA.test];
        testDtA = bsxfun(@minus,testDtA,meanA); testDtA = testDtA*coeff;
        
        testDtB = [squeeze(scncd_nfeat_img.dataB(params.idxtest,h,:)),squeeze(scncd_nfeat_fore.dataB(params.idxtest,h,:))]; 
        cn_typeBasedB.test = [testDtB,cn_typeBasedB.test];
        testDtB = bsxfun(@minus,testDtB,meanB); testDtB = testDtB*coeff;
        
        M.coeff = coeff;
        M.meanA = meanA;  M.meanB = meanB;
        M.testA = testDtA; M.testB = testDtB;
        M.model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
        M.name = sprintf('SCNCD %s',params.cn{h}); 
        
        mSCNCD{h} = M;
    end
    
    % Type-Based Model Evaluation
    coeff = pca([cn_typeBasedA.train;cn_typeBasedB.train]);
    % Projecting data at a low-dimensional space
    coeff = coeff(:,1:params.Type_Factors);
    meanA = mean(cn_typeBasedA.train); 
    nDtA = bsxfun(@minus,cn_typeBasedA.train,meanA); nDtA = nDtA*coeff;
        
    meanB = mean(cn_typeBasedB.train); 
    nDtB = bsxfun(@minus,cn_typeBasedB.train,meanB); nDtB = nDtB*coeff;
    
    % Same individuals
    dataSame = bsxfun(@minus,nDtA,nDtB); 
    % Normalization (zero mean)
    dataSame = bsxfun(@minus,dataSame,mean(dataSame));

    % Not same individuals
    dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
    % Normalization    
    dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
    
    testDtA = bsxfun(@minus,cn_typeBasedA.test,meanA); testDtA = testDtA*coeff;
    testDtB = bsxfun(@minus,cn_typeBasedB.test,meanB); testDtB = testDtB*coeff;
    
    typeRcn=[];
    model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
    [~,typeRcn] = sort(sqdist(testDtA',testDtB',model),2,'ascend');
    
    for i=1:size(typeRcn,2)
        resp_type_scncd(iter, typeRcn(i,:)==i) = resp_type_scncd(iter, typeRcn(i,:)==i) + 1; 
    end
    
    % Evaluating the histograms
    rank_hist= {};
    for n=1:numel(mHist)
        disp(sprintf('Evaluate the Histogram Model (%s)',mHist{n}.name));
        idx=[];
        [~,idx] = sort(sqdist(mHist{n}.testA',mHist{n}.testB',mHist{n}.model),2,'ascend');
        rank_hist{n} = idx;
        for i=1:size(idx,2)
            response_hist(n,iter,idx(i,:)==i) = response_hist(n,iter,idx(i,:)==i) + 1; 
        end
        
    end
    
    rank_scncd={};
    for n=1:numel(mSCNCD)
        disp(sprintf('Evaluate the Histogram Model (%s)',mSCNCD{n}.name));
        idx=[];
        [~,idx] = sort(sqdist(mSCNCD{n}.testA',mSCNCD{n}.testB',mSCNCD{n}.model),2,'ascend');
        rank_scncd{n} = idx;
        for i=1:size(idx,2)
            response_scncd(n,iter,idx(i,:)==i) = response_scncd(n,iter,idx(i,:)==i) + 1; 
        end
    end
    
    
    % FUSION OF COLOR NAMES AND HISTOGRAMS.
    Dta = [cn_typeBasedA.train, hist_typeBasedA.train];
    Dtb = [cn_typeBasedB.train, hist_typeBasedB.train];
    
    % Fusion
    coeff = pca([Dta;Dtb]);   
    % Projecting data at a low-dimensional space
    coeff = coeff(:,1:params.Fusion); 
    meanA = mean(Dta); 
    nDtA = bsxfun(@minus,Dta,meanA); nDtA = nDtA*coeff;
        
    meanB = mean(Dtb); 
    nDtB = bsxfun(@minus,Dtb,meanB); nDtB = nDtB*coeff;
    
    % Same individuals
    dataSame = bsxfun(@minus,nDtA,nDtB); 
    % Normalization (zero mean)
    dataSame = bsxfun(@minus,dataSame,mean(dataSame));

    % Not same individuals
    dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
    % Normalization    
    dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
    
    tDta = [cn_typeBasedA.test, hist_typeBasedA.test];
    tDtb = [cn_typeBasedB.test, hist_typeBasedB.test];
    
    testDtA = bsxfun(@minus,tDta,meanA); testDtA = testDtA*coeff;
    testDtB = bsxfun(@minus,tDtb,meanB); testDtB = testDtB*coeff;
    
    model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
    fusion=[];
    [~,fusion] = sort(sqdist(testDtA',testDtB',model),2,'ascend');
    
    for i=1:size(fusion,2)
        response_fusion(iter,fusion(i,:)==i) = response_fusion(iter,fusion(i,:)==i) + 1; 
    end
   
    % Aggr. of Histograms.
    aggr_hist = zeros(numel(params.idxtest),numel(params.idxtest));
    for i=1:numel(params.idxtest)
        % Aggregation of Ranking Lists
        list={};
        for n=1:numel(mHist)
            list{end+1} = rank_hist{n}(i,:); 
        end
        list{end+1} = typeRhist(i,:);
        
        [~, pval, ~] = aggregateRanks(list,numel(params.idxtest),params.aggr_method,0);
        [~, aggr_hist(i,:)] = sort(pval);
        response_aggr_hist(iter,aggr_hist(i,:)==i) = response_aggr_hist(iter, aggr_hist(i,:)==i) + 1; 
    end

    % Aggr. of Color Names
    aggr_cn = zeros(numel(params.idxtest),numel(params.idxtest));
    for i=1:numel(params.idxtest)
        % Aggregation of Ranking Lists
        list={};
        for n=1:numel(mSCNCD)
            list{end+1} = rank_scncd{n}(i,:); 
        end
        list{end+1} = typeRcn(i,:);
        
        [~, pval, ~] = aggregateRanks(list,numel(params.idxtest),params.aggr_method,0);
        [~, aggr_cn(i,:)] = sort(pval);
        response_aggr_cn(iter, aggr_cn(i,:)==i) = response_aggr_cn(iter, aggr_cn(i,:)==i) + 1; 
    end
  
    % Final Aggregation of color names, histograms and fusion.
    for i=1:numel(params.idxtest)
        % Aggregation of Ranking Lists
        list={};
        list{end+1} = aggr_cn(i,:); 
        list{end+1} = aggr_hist(i,:); 
        list{end+1} = fusion(i,:);
        
        [~, pval, ~] = aggregateRanks(list,numel(params.idxtest),params.aggr_method,0);
        [~, aggr_final] = sort(pval);
        response_final(iter,aggr_final==i) = response_final(iter, aggr_final==i) + 1; 
    end 
end
    % Histogram mean results.
    answer_hist = {};legend_hist={};
    for i=1:numel(params.hist)
        answer_hist{end+1} = cumsum(squeeze(mean(response_hist(i,:,:))))./numel(params.idxtest);
        legend_hist{end+1} = mHist{i}.name;
    end
    answer_hist{end+1} = cumsum(mean(resp_type_hist))./numel(params.idxtest);
    legend_hist{end+1} = sprintf('Hist. Type-Based');   
    
    answer_hist{end+1} = cumsum(mean(response_aggr_hist))./numel(params.idxtest);
    legend_hist{end+1} = 'Aggr. Hist';
    
    % Color Names mean results.
    answer_cn = {};legend_cn={};
    for i=1:numel(params.cn)
        answer_cn{end+1} = cumsum(squeeze(mean(response_scncd(i,:,:))))./numel(params.idxtest);
        legend_cn{end+1} = mSCNCD{i}.name;
    end
    answer_cn{end+1} = cumsum(mean(resp_type_scncd))./numel(params.idxtest);
    legend_cn{end+1} = sprintf('SCNCD Type-Based'); 
    
    answer_cn{end+1} = cumsum(mean(response_aggr_cn))./numel(params.idxtest);
    legend_cn{end+1} = sprintf('Aggr. SCNCD'); 
    
    answer={};legend={};
    %Cascade Results.
    answer{end+1} = cumsum(mean(response_fusion))./numel(params.idxtest);
    legend{end+1} = sprintf('Fusion');
    
    answer{end+1} = cumsum(mean(response_final))./numel(params.idxtest);
    legend{end+1} = 'Aggr.';
    
    % Ploting histograms, fusion and type-based features.
    fileName = 'Histograms ICIP 2015';
    ntitle = sprintf('CMC Curve - %s',upper(params.dataset));
    PlotCurve(answer_hist, params.saveDir, fileName, legend_hist, ntitle); 
    set(gcf,'color','w');
    
    
    % Ploting color names, fusion and type-based features.
    fileName = 'Color Names ICIP 2015';
    ntitle = 'CMC Curve - VIPER';
    PlotCurve(answer_cn, params.saveDir, fileName, legend_cn, ntitle); 
    set(gcf,'color','w');
    
    % Ploting the aggregation and fusion of color names and histograms 
    fileName = 'Cascade ICIP 2015';
    ntitle = 'CMC Curve - VIPER';
    PlotCurve(answer, params.saveDir, fileName, legend, ntitle); 
    set(gcf,'color','w');
end

%Extracting the color names descriptor for Img representation
function nfeatures = getSCNCDImg(params, imagesA, imagesB, mask)
indexParts = getIndex(params.image, params.nparts);
type = 'cn'; 
models = params.cn;
for m=1:numel(models)
    for i=1:numel(imagesA)
            if  strcmpi(models{m},'HSV')
                imgA = rgb2hsv(imagesA{i}.img).*256;
                imgB = rgb2hsv(imagesB{i}.img).*256;
                else if strcmpi(models{m},'nrgb')
                    imgA = rgb2norm_rgb(imagesA{i}.img).*256;
                    imgB = rgb2norm_rgb(imagesB{i}.img).*256;
                    else if strcmpi(models{m},'RGB')
                        imgA = imagesA{i}.img.*256;
                        imgB = imagesB{i}.img.*256;
                        else if strcmpi(models{m},'L1L2L3')
                            imgA = rgb2L1L2L3(imagesA{i}.img).*256;
                            imgB = rgb2L1L2L3(imagesB{i}.img).*256;
                            end
                        end
                    end
            end
            % Extracting the features for camera A
            index  = img2index(imgA,[],type);
            aux = [];
            for k=1:params.nparts
              hist = mask(indexParts==k)'*params.colorsTable(index(indexParts==k),:);
              aux = [aux,normr(hist)];   
            end
            featA= normr(aux);
            
            % Extracting the features for camera B
            index  = img2index(imgB,[],type);
            aux = [];
            for k=1:params.nparts
              hist = mask(indexParts==k)'*params.colorsTable(index(indexParts==k),:);
              aux = [aux,normr(hist)];   
            end
            featB = normr(aux);

            nfeatures.dataA(i,m,:)= featA;
            nfeatures.dataB(i,m,:) = featB;
    end
end

end

%Extracting the color names descriptor for Foreground representation
function nfeatures = getSCNCDFore(params, imagesA, imagesB, maskA, maskB)
indexParts = getIndex(params.image, params.nparts);
type = 'cn'; 
models = params.cn;
featA={}; featB={};
for m=1:numel(models)
    for i=1:numel(imagesA)
            if  strcmpi(models{m},'HSV')
                imgA = rgb2hsv(imagesA{i}.img).*256;
                imgB = rgb2hsv(imagesB{i}.img).*256;
                else if strcmpi(models{m},'nrgb')
                    imgA = rgb2norm_rgb(imagesA{i}.img).*256;
                    imgB = rgb2norm_rgb(imagesB{i}.img).*256;
                    else if strcmpi(models{m},'RGB')
                        imgA = imagesA{i}.img.*256;
                        imgB = imagesB{i}.img.*256;
                        else if strcmpi(models{m},'L1L2L3')
                            imgA = rgb2L1L2L3(imagesA{i}.img).*256;
                            imgB = rgb2L1L2L3(imagesB{i}.img).*256;
                            end
                        end
                    end
            end
            % Extracting the features for camera A
            index  = img2index(imgA,[],type);
            aux = [];
            mask = maskA(:,:,i);
            for k=1:params.nparts
              hist = mask(indexParts==k)'*params.colorsTable(index(indexParts==k),:);
              aux = [aux,normr(hist)];   
            end
            featA = normr(aux);
            
            % Extracting the features for camera B
            index  = img2index(imgB,[],type);
            aux = [];
            mask = maskB(:,:,i);
            for k=1:params.nparts
              hist = mask(indexParts==k)'*params.colorsTable(index(indexParts==k),:);
              aux = [aux,normr(hist)];   
            end
            featB = normr(aux);
            
            nfeatures.dataA(i,m,:) = featA;
            nfeatures.dataB(i,m,:) = featB;
    end
end

end

% Converting the RGB values in the index to access the color names table or
% obtaining the bin index for each pixel in histogram.
function index = img2index(img, bins, type)
    if strcmp(type,'cn')
        index= 1+floor(img(:,:,1)./8)+32*floor(img(:,:,2)./8)+32*32*floor(img(:,:,3)./8);
        index(isnan(index)) = 1;
        index((index> 32768)) = 32768;
        index((index<1))= 1;
    else if strcmp(type,'hist')
            aux = 256/bins;
            index= floor(img./aux)+1;
            index((index> bins)) = bins;
            index((index<1))= 1;
        end
    end
end

function nimg = rgb2L1L2L3(img)
% Accordingly to the paper: Color Based Ob ject Recognition ( Gevers and Smeulders)
% l1 = (R-G)� / (R-G)� + (R-B)� + (G-B)�
% l2 = (R-B)� / (R-G)� + (R-B)� + (G-B)�
% l3 = (G-B)� / (R-G)� + (R-B)� + (G-B)�
a1 = (img(:,:,1)-img(:,:,2)).^2 + eps; a2 = (img(:,:,1)-img(:,:,3)).^2 + eps; a3 = (img(:,:,2)-img(:,:,3)).^2 + eps;    
nimg(:,:,1)  = (a1./(a1+a2+a3 )); 
nimg(:,:,2)  = (a2./(a1+a2+a3));  
nimg(:,:,3)  = (a3./(a1+a2+a3 )); 

end

function nimg = rgb2norm_rgb(img)
    % Normalized RGB
    % r = R/(R + G + B)
    % g = G/(R + G + B)
    % b = B/(R + G + B)
    nimg(:,:,1) = (img(:,:,1)./(img(:,:,1) + img(:,:,2) + img(:,:,3)+ eps));
    nimg(:,:,2) = (img(:,:,2)./(img(:,:,1) + img(:,:,2) + img(:,:,3)+ eps));
    nimg(:,:,3) = (img(:,:,3)./(img(:,:,1) + img(:,:,2) + img(:,:,3)+ eps)); 
end

% Reading images from datasets.
function [N,imagesA, imagesB]= read(imageSize,dataset)
    dirnameA = sprintf('./datasets/%s/camA/',dataset); dirnameB = sprintf('./datasets/%s/camB/',dataset);
    if strcmp(dataset,'viper')
        format = '*bmp';
    else if strcmp(dataset,'prid450S')
        format = '*png';
        end
    end
    dd = dir(strcat(dirnameA,format));
    fileNames = {dd.name}; 
    
    imagesA= {};imagesB = {};
    for i=1:numel(fileNames)
        aux = imread(fullfile(dirnameA,fileNames{i})); 
        imagesA{end+1}.img = imresize(im2double(aux), imageSize);
    end

    dd = dir(strcat(dirnameB,format));
    fileNames = {dd.name}; 
    %% Loading imags at camera B
    for i=1:numel(fileNames)
        aux = imread(fullfile(dirnameB,fileNames{i})); 
        imagesB{end+1}.img = imresize(im2double(aux), imageSize);
    end
    N = numel(fileNames);
end

% Computing the Gaussian Mask representation.
function mask_gaus = getGauss(image)
    % Obtaining the mask representation
    mask_gaus = repmat(1:image(2), image(1),1);
    mask_gaus =  exp(-((mask_gaus - size(mask_gaus,2)/2).^2)./(2.*(size(mask_gaus,2)/4).^2));
end

%Indexing the image stripes.
function index_parts = getIndex(imageSize,num_stripes)
    width = round(imageSize(1)/num_stripes);
    index_parts = zeros(imageSize);
    for j=1:num_stripes
        index_parts((j-1)*width +1: j*width, :)=j;
    end
    index_parts(index_parts==0) = num_stripes;
end

% getHist receives the images from camera A (imagesA) and B (imagesB) and
% the color models. The output is a 3D matrix with image_id,models,
% feature descriptor.
function nfeatures = getHistImg(params,imagesA, imagesB, mask)
indexParts = getIndex(params.image, params.nparts);
type = 'hist';
models = params.hist;
%% Avaliating features for person Re-Identification
for m=1:numel(models)
    for i=1:numel(imagesA)
            if  strcmpi(models{m},'HSV')
                imgA = rgb2hsv(imagesA{i}.img).*256;
                imgB = rgb2hsv(imagesB{i}.img).*256;
                else if strcmpi(models{m},'nrgb')
                    imgA = rgb2norm_rgb(imagesA{i}.img).*256;
                    imgB = rgb2norm_rgb(imagesB{i}.img).*256;
                    else if strcmpi(models{m},'RGB')
                        imgA = imagesA{i}.img.*256;
                        imgB = imagesB{i}.img.*256;
                        else if strcmpi(models{m},'L1L2L3')
                            imgA = rgb2L1L2L3(imagesA{i}.img).*256;
                            imgB = rgb2L1L2L3(imagesB{i}.img).*256;
                            end
                        end
                    end
            end

            %Computing the histogram to image at camera A     
            featA={};
            for j=1:size(imgA,3)    
                index  = img2index(imgA(:,:,j), params.hist_bins, type);
                aux = [];
                for k=1:params.nparts    
                  aux = [aux,maskHist(indexParts, index, mask, params.hist_bins, k)];   
                end
                featA{end+1} = aux; 
            end
            %Computing the histogram to image at camera B
            featB={};
            for j=1:size(imgB,3)    
                index  = img2index(imgB(:,:,j), params.hist_bins, type);
                aux = [];
                for k=1:params.nparts    
                  aux = [aux,maskHist(indexParts, index, mask, params.hist_bins, k)];   
                end
                featB{end+1} = aux; 
            end
            nfeatures.dataA(i,m,:)= normr(cell2mat(featA));
            nfeatures.dataB(i,m,:) = normr(cell2mat(featB));
    end
    
end
end

% getHist receives the images from camera A (imagesA) and B (imagesB) and
% the color models. The output is a 3D matrix with image_id,models,
% feature descriptor.
function nfeatures = getHistFore(params,imagesA, imagesB, maskA, maskB)
indexParts = getIndex(params.image, params.nparts);
type = 'hist';
models = params.hist;
%% Avaliating features for person Re-Identification
for m=1:numel(models)
    for i=1:numel(imagesA)
            if  strcmpi(models{m},'HSV')
                imgA = rgb2hsv(imagesA{i}.img).*256;
                imgB = rgb2hsv(imagesB{i}.img).*256;
                else if strcmpi(models{m},'nrgb')
                    imgA = rgb2norm_rgb(imagesA{i}.img).*256;
                    imgB = rgb2norm_rgb(imagesB{i}.img).*256;
                    else if strcmpi(models{m},'RGB')
                        imgA = imagesA{i}.img.*256;
                        imgB = imagesB{i}.img.*256;
                        else if strcmpi(models{m},'L1L2L3')
                            imgA = rgb2L1L2L3(imagesA{i}.img).*256;
                            imgB = rgb2L1L2L3(imagesB{i}.img).*256;
                            end
                        end
                    end
            end

            %Computing the histogram to image at camera A     
            featA={};
            for j=1:size(imgA,3)    
                index  = img2index(imgA(:,:,j), params.hist_bins, type);
                aux = [];
                for k=1:params.nparts    
                  aux = [aux,maskHist(indexParts, index, maskA(:,:,i), params.hist_bins, k)];   
                end
                featA{end+1} = aux; 
            end
            %Computing the histogram to image at camera B
            featB={};
            for j=1:size(imgB,3)    
                index  = img2index(imgB(:,:,j), params.hist_bins, type);
                aux = [];
                for k=1:params.nparts    
                  aux = [aux,maskHist(indexParts, index, maskB(:,:,i), params.hist_bins, k)];   
                end
                featB{end+1} = aux; 
            end
    end
    nfeatures.dataA(i,m,:)= normr(cell2mat(featA));
    nfeatures.dataB(i,m,:) = normr(cell2mat(featB));
end
end


function color_hist = maskHist(index_parts, index, mask, bins, id_part)
    color_hist = zeros(1,bins);
    for i=1:bins
        color_hist(i) = sum(mask(index_parts==id_part & index==i));
    end
    color_hist = normr(color_hist);
end


function [mskA, mskB] = splitMasks(msk)
    mskA = []; mskB=[]; 
    id=1;
    for i=1:2:numel(msk)
        mskA(:,:,id) = imfill(msk{i},'holes'); mskB(:,:,id) = imfill(msk{i+1},'holes');
        id = id + 1;
    end
end
