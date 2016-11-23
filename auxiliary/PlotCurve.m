function PlotCurve(cmcs, saveDir, dataset, plots, ntitle)
% squared-black blue-ball red-diamond green-cross
color={[0,0,0],[0,0,1],[1,0,0],[0.1,0.8,0.1],[0.5,0,0.5],[0.3,0.75,0.9],[0,0.5,0.5],[0.5,0,0.5],[0.5,0,0],[0,0.5,0]};
Markers = {'s', 'o','d', 'x','^','*','<','>','v'};

figure('Units', 'pixels', ...
    'Position', [100 100 500 375]);
    hold on;

for i=1:numel(cmcs)
    formatSpec = strcat(plots{i},' (%.1f%%)');
    rank_1(i) = cmcs{i}(1)*100;
    legends{i}= sprintf(formatSpec,rank_1(i));
    disp(legends{i});
    disp(sprintf('Rank-1 : %.1f%% ', cmcs{i}(1)*100));
end

[~,idx] =sort(rank_1,'descend');
xdata =[1:numel(cmcs{1})];
nlegends={};
h = figure;
% ridx =[1,5,10,15,20,25,30,35,40,45,50];
ridx =[1,5,10,15,20,25,30,35,40,45,50];
for i=1: numel(cmcs)
    plot(xdata(ridx), ...
        cmcs{idx(i)}(ridx),...
        'LineWidth',2.0,...
        'Color',color{i},...
        'Marker',Markers{i});
    hold on;
    nlegends{i}= legends{idx(i)};
end


yTitle= ylabel('Matching Rate');
xTitle= xlabel('Rank Position');
hTitle = title(ntitle);

ylim([0 1]);

set([xTitle,yTitle], ...
   'FontName'   , 'AvantGarde',...
   'FontSize'   , 10);

set(hTitle, 'FontWeight' , 'bold',...
            'FontName'   , 'AvantGarde',...
            'FontSize'   , 12);

set(gca,'TickLength', [0 0],...
        'FontSize', 10);


legend(upper(nlegends),...
       'Location','SouthEast',...
       'FontName'   , 'Helvetica',...
       'FontSize', 12);

legend boxoff
% 
% mkdir(saveDir);
% for i=1:numel(cmcs)
%     dlmwrite(sprintf('%s%s%d.txt',saveDir,nlegends{i},i),cmcs{i}');
% end

set(gcf, 'PaperPositionMode', 'auto');
print(sprintf('%scmc_curve_%s.eps',saveDir,dataset),'-depsc2');
exportAndCropFigure(h,dataset,saveDir);


