% a simple matlab script to compute AUC

data = load('output.csv');

[X,Y,t,AUC] = perfcurve(data(:,1),data(:,3),'1');

figure(1); clf;
plot(X,Y)
title(AUC)
xlabel('FPR')
ylabel('TPR')
AUC

