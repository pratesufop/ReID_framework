workDir = pwd;
% creating data dir
if  exist(fullfile(workDir,'data'))==0
    mkdir(fullfile(workDir,'data'));
end

if exist(fullfile(workDir,'data','GOG_VIPeR.zip'))==0
    websave(fullfile(workDir,'data','GOG_VIPeR.zip'),'http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/GOG_VIPeR.zip')
    unzip(fullfile(workDir,'data','GOG_VIPeR'),fullfile(workDir,'data'));
    matfiles = dir(fullfile('data','GOG_VIPeR','*.mat'));
    matfiles = {matfiles.name};
    features =[];
    for i=1:numel(matfiles)
        auxDt = load(fullfile(workDir,'data','GOG_VIPeR',matfiles{i}));
        features = [features,auxDt.feature_all];
        clear auxDt
    end
    save(fullfile(workDir,'data','viper_features.mat'),'features');
end
