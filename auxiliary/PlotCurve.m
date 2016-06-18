function PlotCurve(cmcs, saveDir, dataset, plots, ntitle)
% squared-black blue-ball red-diamond green-cross
color={[0,0,0],[0,0,1],[1,0,0],[0.8,0.3,0],[0.5,0,0.5],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0,0],[0,0.5,0]};
Markers = {'s', 'o','d', 'x','^','*','<','>','v'};

Xmax = 50;

figure('Units', 'centimeters', ...
    'Position', [100 100 500 400]);
    hold on;

for i=1:numel(cmcs)
    formatSpec = strcat(plots{i},' (%.2f)');
    area(i) = trapz( [1:Xmax]/Xmax , cmcs{i}(1:Xmax));
    legends{i}= sprintf(formatSpec,area(i));
    disp(legends{i});
    disp(sprintf('Rank-1 : %.2f ', cmcs{i}(1)));
end

[~,idx] =sort(area,'descend');
xdata =[1:numel(cmcs{1})];
nlegends={};
h = figure;
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

yTitle= ylabel('Recognition Rate');
xTitle= xlabel('Ranks');
hTitle = title(ntitle);
set([xTitle,yTitle], ...
   'FontName'   , 'AvantGarde',...
   'FontSize'   , 16);

set(hTitle, 'FontWeight' , 'bold',...
            'FontName'   , 'AvantGarde',...
            'FontSize'   , 14);

set(gca,'TickLength', [0 0],...
        'FontSize', 10);
ylim([0 1]);

legend(upper(nlegends),...
       'Location','SouthEast',...
       'FontName'   , 'Helvetica',...
       'FontSize', 14);

legend boxoff

mkdir(saveDir);
for i=1:numel(cmcs)
    dlmwrite(sprintf('%s%s%d.txt',saveDir,nlegends{i},i),cmcs{i}');
end

set(gcf, 'PaperPositionMode', 'auto');
print -depsc2 finalPlot1.eps

exportAndCropFigure(h,dataset,saveDir);


