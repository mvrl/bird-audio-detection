
clear

%% load data

runs = dir('../checkpoint/*_*/output.csv');
runs = ...
  cellfun(@(x) x((find(x == '/',1,'last')+1):end), ...
  {runs.folder},'UniformOutput',false);

% results across all datasets
eval_all = cell(numel(runs),1);

success = false(size(runs));

for irun = 1:numel(runs)
  
  try
    run = runs{irun};
    log = @(s) fprintf('%s: %s\n',run,s);
    
    
    % WARNING: only using a subset of the testing data
    result_file = ['../checkpoint/' run '/output.csv'];
    log(['loading data (' result_file ')'])
    data = dlmread(result_file,' ',[0,0,1550,2]);
    
    %% make 1-vs-all ROC curves
    
    labels = data(:,1);
    scores = data(:,2:end);
    
    tmp = struct('run',run);
    
    log('computing perfcurve')
    [tmp.X,tmp.Y,tmp.T,tmp.AUC] = perfcurve(labels,scores(:,2),1);
    eval_all{irun} = tmp;
    
    fprintf('AUC = %f\n',tmp.AUC)
    success(irun) = true;
  catch
    fprintf('Failed to load/process properly.')
  end
  
end

runs = runs(success);
eval_all = eval_all(success);

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
  runs(idx), eval_all(idx)','UniformOutput', false);
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

size(runs)