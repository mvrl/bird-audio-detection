addpath ~/matlab_root/

%% load data
runs = {'elu_eeg', 'elu_piezo'};
names = {'WAKE','NREM','REM'};

eval = cell(numel(runs),numel(names));

for irun = 1:numel(runs)
  
  disp('loading data')
  run = runs{irun};
  
  % WARNING: only using a subset of the testing data
  
  data = dlmread(['checkpoint/' run '/output.csv'],' ',[0,0,200000,4]);
  
  good = data(:,end-1) == data(:,end);
  scores = data(good,1:end-2);
  labels = data(good,end);
  
  %% make 1-vs-all ROC curves
  
  
  for iname = 1:numel(names)
    
    disp('computing perfcurve')
    [tmp.X,tmp.Y,tmp.T,tmp.AUC] = perfcurve(labels,scores(:,iname),iname);
    
    eval{irun,iname} = tmp;
    
  end
  
end

%%

figure(1);clf

for iname = 1:numel(names)
  subplot(1,3,iname)
  for irun = 1:numel(runs)
    tmp = eval{irun,iname};
    plot(tmp.X,tmp.Y)
    hold on
  end
  axis square
  grid on
  legend_text = cellfun(...
    @(name,eval) sprintf('%s (%02.4f)',name,eval.AUC), ...
    runs, eval(:,iname)','UniformOutput', false);
  legend(legend_text,'Interpreter','none','Location','southeast')
  set(gca,'XTick', linspace(0,1,5))
  set(gca,'YTick', linspace(0,1,5))
  xlabel('False positive rate')
  ylabel('True positive rate')
  title(['ROC (' names{iname} ')'])
  
end

disp('exporting figure')
exportfigure(gcf,'roc.pdf',[10,3.5])
