clear

%% load data
runs = dir('../checkpoint/*_*');
runs = {runs.name};

% results across all datasets
eval_all = cell(numel(runs),1);

for irun = 1:numel(runs)
  
  run = runs{irun};
  log = @(s) fprintf('%s: %s\n',run,s);
  
  
  % WARNING: only using a subset of the testing data
  log('loading data')
  data = dlmread(['checkpoint/' run '/output.csv'],' ',[0,0,1550,2]);
  
  %% make 1-vs-all ROC curves
  
  labels = data(:,1);
  scores = data(:,2:end);
  
  tmp = struct('run',run);
  
  log('computing perfcurve')
  [tmp.X,tmp.Y,tmp.T,tmp.AUC] = perfcurve(labels,scores(:,2),1);
  
  eval_all{irun} = tmp;
  
end

%%

% sort methods by min AUC
[~,idx] = sort(min(cellfun(@(eval) eval.AUC,eval_all),[],2),'descend');

figure(1);clf

for irun = 1:numel(runs)
  tmp = eval_all{idx(irun)};
  plot(tmp.X,tmp.Y)
  hold on
end
axis square
grid on
legend_text = cellfun(...
  @(name,eval) sprintf('%s (%02.4f)',name,eval.AUC), ...
  runs, eval_all(idx)','UniformOutput', false);
legend(legend_text,'Interpreter','none','Location','southeast')
set(gca,'XTick', linspace(0,1,5))
set(gca,'YTick', linspace(0,1,5))
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC')

disp('exporting figure')
exportfigure(gcf,'roc.pdf',[12,5.5])
savefig(gcf,'roc.fig')

save('performance.mat','runs','eval_all')

