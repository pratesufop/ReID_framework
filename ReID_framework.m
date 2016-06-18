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
ICB(dataset);
% To run our ICIP2016 code.
% ICIP2016
