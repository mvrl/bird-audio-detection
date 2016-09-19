addpath ~/matlab_root/

%% load data
%runs = {'elu_piezo', 'elu_eeg', 'relu_piezo', 'relu_eeg', 'elu_piezo_2', 'elu_eeg_2', 'relu_piezo_2', 'relu_eeg_2'};
runs = dir('checkpoint'); runs = {runs(3:end).name};
names = {'WAKE','NREM','REM'};

% results across all datasets
eval_all = cell(numel(runs),numel(names));

for irun = 1:numel(runs)
  
  run = runs{irun};
  log = @(s) fprintf('%s: %s\n',run,s);
  
  
  % WARNING: only using a subset of the testing data
  log('loading data')
  data = dlmread(['checkpoint/' run '/output.csv'],' ',[0,0,200000,6]);
  
  %% make 1-vs-all ROC curves
  
  good = data(:,end-1) == data(:,end);
  fileids = data(good,1);
  rows = data(good,2);
  scores = data(good,3:end-2);
  labels = data(good,end);
  
  for iname = 1:numel(names)
    
    log('computing perfcurve')
    [tmp.X,tmp.Y,tmp.T,tmp.AUC] = perfcurve(labels,scores(:,iname),iname);
    
    eval_all{irun,iname} = tmp;
    
  end
  
end

%%

figure(1);clf

for iname = 1:numel(names)
  subplot(1,3,iname)
  for irun = 1:numel(runs)
    tmp = eval_all{irun,iname};
    plot(tmp.X,tmp.Y)
    hold on
  end
  axis square
  grid on
  legend_text = cellfun(...
    @(name,eval) sprintf('%s (%02.4f)',name,eval.AUC), ...
    runs, eval_all(:,iname)','UniformOutput', false);
  legend(legend_text,'Interpreter','none','Location','southeast')
  set(gca,'XTick', linspace(0,1,5))
  set(gca,'YTick', linspace(0,1,5))
  xlabel('False positive rate')
  ylabel('True positive rate')
  title(['ROC (' names{iname} ')'])
  
end

disp('exporting figure')
exportfigure(gcf,'roc.pdf',[10,3.5])
