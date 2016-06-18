function ICIP2016

% Loading the color names and the foreground masks
load('auxiliary\trCoe.mat'); 
% PARAMETERS
% Dataset (VIPer or PRID450S)
params.dataset = 'viper';
params.colorsTable = w2c;
params.nIter = 10;
% image size and the number of parts.
params.image = [128,48];
params.nparts = 6;
params.cn ={'RGB'};

params.factors = 50;
% Load partition
load(sprintf('auxiliary\\Partition_%s.mat',params.dataset));
params.saveDir = strcat('.\Graphics\ICIP2016\',params.dataset,'\');

% Load the images and Masks(VIPER and PRID450S datasets). 
% N corresponds to the number of individuals 
[N, imgA, imgB] = read(params.image,params.dataset);
% Gaussian Mask
params.Gauss_mask = getGauss(params.image);

% Extract Color Names Features.
scncd_nfeat = getSCNCDImg(params, imgA, imgB, params.Gauss_mask);
% These features are used to evaluate the indexation
% They are our implementation of the SCNCD 
data_features = open(sprintf('./auxiliary\\%s_features_scncd.mat',params.dataset));

% loading the features
dataA=[]; dataB=[];
for i=1:N
    dataA(i,:) = [data_features.features(:,i)]';
    dataB(i,:) = [data_features.features(:,size(data_features.features,2)/2+i)]';
end

% Response for all curves.
resp_baseline = zeros(params.nIter,N/2);
respRandom = zeros(params.nIter,N/2);
resp50 = zeros(params.nIter,N/2);
resp25 = zeros(params.nIter,N/2);
resp75 = zeros(params.nIter,N/2);

params.num_pcn = 3;

% store values F and S accordingly with the paper.
params.F = zeros(params.nparts, size(params.colorsTable,2), N/2);
params.S = zeros(params.nparts, size(params.colorsTable,2), N/2);

for iter=1:params.nIter
    % Learn the metric distance (KISSME).
    % Color Names descriptor
    params.idxtrain = trials(iter,1:N/2);
    params.idxtest = trials(iter,N/2 + 1:end);
    notsame = randperm(N/2,N/2);
    
    % Indexing using Predominant Color Names
    galleryDt = scncd_nfeat.dataB(params.idxtest,:);
    probeDt = scncd_nfeat.dataA(params.idxtest,:);
    % filling the indexing structure (params.indexed)
    indexed = PCN(galleryDt,probeDt,params);
    
    % Data from camera A
    nDtA = dataA(params.idxtrain,:); 
    
    % Data from camera B
    nDtB = dataB(params.idxtrain,:); 
    
     % Same individuals
    dataSame = bsxfun(@minus,nDtA,nDtB); 
    % Normalization (zero mean)
    dataSame = bsxfun(@minus,dataSame,mean(dataSame));

    % Not same individuals
    dataNotSame = bsxfun(@minus,nDtA,nDtB(notsame,:));
    % Normalization    
    dataNotSame = bsxfun(@minus,dataNotSame,mean(dataNotSame)); 
    
    % KISS Metric Learning Model
    KissML.model = inv(dataSame'*dataSame) - inv(dataNotSame'*dataNotSame);
    
    testDta = dataA(params.idxtest,:); 
    testDtb = dataB(params.idxtest,:);
    
    [~,rank_kiss] = sort(sqdist(testDta',testDtb',KissML.model),2,'ascend');
    
    % Baseline: Comparing probe against the entire gallery (no-indexing).
    resp = zeros(1,numel(params.idxtest));
    for i=1:numel(params.idxtest)
        resp(rank_kiss(i,:)==i) = resp(rank_kiss(i,:)==i) + 1; 
    end
    resp_baseline(iter,:) = cumsum(resp)./numel(params.idxtest);     
    
    % Indexing @0.75
    rank=[];
    pos = round(0.75*numel(params.idxtest));
    for i=1:numel(params.idxtest)
        candidates = indexed{i}; candidates = candidates(1:min(pos,numel(candidates))); 
        dist = sqdist(testDta(i,:)',testDtb(candidates,:)',KissML.model);
        [~,idx] = sort(dist,2,'ascend'); 
        rank(i,:) = candidates(idx);
    end
    % Response using 75% of gallery images
    resp = zeros(1,numel(params.idxtest));
    for i=1:numel(params.idxtest)
        resp(rank(i,:)==i) = resp(rank(i,:)==i) + 1; 
    end
    resp75(iter,:) = cumsum(resp)./numel(params.idxtest); 
    
    % Indexing @0.50
    rank=[];
    pos = round(0.50*numel(params.idxtest));
    for i=1:numel(params.idxtest)
        candidates = indexed{i}; candidates = candidates(1:min(pos,numel(candidates))); 
        dist = sqdist(testDta(i,:)',testDtb(candidates,:)',KissML.model);
        [~,idx] = sort(dist,2,'ascend'); 
        rank(i,:) = candidates(idx);
    end
    % Response using 50% of gallery images
    resp = zeros(1,numel(params.idxtest));
    for i=1:numel(params.idxtest)
        resp(rank(i,:)==i) = resp(rank(i,:)==i) + 1; 
    end
    resp50(iter,:) = cumsum(resp)./numel(params.idxtest); 
    
    % Indexing @0.25
    rank=[];
    pos = round(0.25*numel(params.idxtest));
    for i=1:numel(params.idxtest)
        candidates = indexed{i}; candidates = candidates(1:min(pos,numel(candidates))); 
        dist = sqdist(testDta(i,:)',testDtb(candidates,:)',KissML.model);
        [~,idx] = sort(dist,2,'ascend'); 
        rank(i,:) = candidates(idx);
    end
    % Response using 25% of gallery images
    resp = zeros(1,numel(params.idxtest));
    for i=1:numel(params.idxtest)
        resp(rank(i,:)==i) = resp(rank(i,:)==i) + 1; 
    end
    resp25(iter,:) = cumsum(resp)./numel(params.idxtest); 
    
    % Random
    rank=[];
    pos = round(0.5*numel(params.idxtest));
    for i=1:numel(params.idxtest)
        candidates = randperm(numel(params.idxtest),numel(params.idxtest));
        candidates = candidates(1:min(pos,numel(candidates))); 
        
        dist = sqdist(testDta(i,:)',testDtb(candidates,:)',KissML.model);
        [~,idx] = sort(dist,2,'ascend'); 
        rank(i,:) = candidates(idx);
    end
    % Response using 25% of gallery images
    resp = zeros(1,numel(params.idxtest));
    for i=1:numel(params.idxtest)
        resp(rank(i,:)==i) = resp(rank(i,:)==i) + 1; 
    end
    respRandom(iter,:) = cumsum(resp)./numel(params.idxtest); 
    
end
% Baseline Result.
answer = {};legend={};

% Baseline - KISSME (no-indexing)
answer{end+1} = mean(resp_baseline,1);
legend{end+1} = sprintf('No Indexing @1.0'); 

answer{end+1} = mean(resp75,1);
legend{end+1} = sprintf('Indexing @0.75'); 

answer{end+1} = mean(resp50,1);
legend{end+1} = sprintf('Indexing @0.50'); 

answer{end+1} = mean(resp25,1);
legend{end+1} = sprintf('Indexing @0.25');

answer{end+1} = mean(respRandom,1);
legend{end+1} = sprintf('No Indexing @0.50'); 

% Ploting the graphic in Fig. 6
fileName = 'Performance Degradation';
ntitle = sprintf('CMC Curve - %s',upper(params.dataset));
PlotCurve(answer, params.saveDir, fileName, legend, ntitle); 
set(gcf,'color','w');

end

% Indexing images using the predominant color names
function indexed = PCN(gallery,probe, params)
    indexed={};
    num_cn = size(params.colorsTable,2);
    for n=1:size(gallery,1)
        fprintf('PCN for Gallery Image %d\n',n);
        feat = gallery(n,:); feat = reshape(feat,params.nparts,num_cn);
        for k=1:params.nparts
            [vx,idx] = sort(feat(k,:),'descend');
            params.F(k,idx(1:params.num_pcn),n)=1;
            params.S(k,idx(1:params.num_pcn),n)=vx(1:params.num_pcn);
        end
    end
    
    for n=1:size(probe,1)
        fprintf('Indexing Probe Image %d\n',n);
        feat = probe(n,:); feat = reshape(feat,params.nparts,num_cn);
        candidates ={};
        for k=1:params.nparts
            [~,idx] = sort(feat(k,:),'descend');
            aux = find(sum(params.F(k,idx(1:params.num_pcn),:))>=1);
            [~,idx] = sort(sum(params.S(k,idx(1:params.num_pcn),aux)),'descend');
            missing =find(~ismember([1:numel(params.idxtest)],squeeze(aux(idx))));
            idm = randperm(numel(missing),numel(missing));
            % We need to fill the list to use Borda Count method
            candidates{end+1} = [squeeze(aux(idx));missing(idm)'];
        end
        indexed{n} = Borda_Count(candidates) ;
    end
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
% l1 = (R-G)² / (R-G)² + (R-B)² + (G-B)²
% l2 = (R-B)² / (R-G)² + (R-B)² + (G-B)²
% l3 = (G-B)² / (R-G)² + (R-B)² + (G-B)²
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

function nrank =  Borda_Count(ranks)
if numel(ranks)>1
    ids= unique(ranks{1});
    score = zeros(1,numel(ids));
    for i=1:numel(ids)
        for j=1:numel(ranks)
            score(i) = score(i) + find(ids(i) == ranks{j});
        end
    end

    [~,idx] = sort(score,'ascend');
    nrank = ids(idx);
else
    nrank= ranks{1};
end

end