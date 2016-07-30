function ReID_framework
% Re-ID framework developed by 
% Raphael Prates and William Robson Schwartz (UFMG-Brazil)
% Any questions, please contact us in: 
% Raphael Prates (pratesufop@gmail.com).

close all

addpath '.\KISSME'
addpath '.\auxiliary'
addpath '.\ranking_aggregation'

dataset = 'viper'; 
% dataset='prid450S';

% To run our ICIP 2015 code (coined CBRA), uncomment the following line
% ICIP(dataset);

% To run our ICB code (Prototypes-based method), uncomment the following line
%ICB(dataset);

% To run our ICIP2016 code.
% ICIP2016

% OBS for AVSS2016 and ICPR2016 code.
% Until now, we only use features from viper and prid450s.
% To run for other datasets, insert the features in auxiliary folder and do
% the modifications in the code. Notice that you need to create a partition
% and save jointly with data.

% To run our AVSS2016 code.
filename = 'myGraphAVSS2016';
AVSS2016(filename, dataset)

%To run our ICPR2016 code.
%ICPR2016(filename, dataset)
