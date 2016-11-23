workDir = pwd;
if exist(fullfile(workDir,'data','GOG_CUHK01.zip'))==0
    websave(fullfile(workDir,'data','GOG_CUHK01.zip'),'http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/GOG_CUHK01.zip')
    unzip(fullfile(workDir,'data','GOG_CUHK01.zip'),fullfile(workDir,'data'));
    matfiles = dir(fullfile('data','GOG_CUHK01','*.mat'));
    matfiles = {matfiles.name};
    features =[];
    for i=1:numel(matfiles)
        auxDt = load(fullfile(workDir,'data','GOG_CUHK01',matfiles{i}));
        auxDt.feature_all = normr(bsxfun(@minus,auxDt.feature_all, mean(auxDt.feature_all,2)));
        features = [features,auxDt.feature_all];
        clear auxDt
    end
    save(fullfile(workDir,'data','cuhk01_features.mat'),'features');
end