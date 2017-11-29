
gbm = 77.9;
rf = 77.6;
ada = 77.4;
et = 77.2;
svm = 77.4;
ann = 77.1;

accuracy = [gbm, rf, ada, et, svm, ann];

NumTicks = 6;
bar(1:NumTicks, accuracy)
L = get(gca,'XLim');
set(gca,'XTick', 1:NumTicks, 'xTickLabel', {'GBM', 'RF', 'AdaBoost', 'ET', 'SVM', 'ANN'})
ylim([76, 79])
xlabel('Models')
ylabel('Validation Accuracy (%)')
title('Model selection')

x = 200; 
y = 200; 
width = 600; 
height = 400; 
set(figure(1), 'Position', [x y width height])
a = findobj(gcf); 
allaxes = findall(a, 'Type', 'axes'); 
alllines = findall(a, 'Type', 'line'); 
set(alllines, 'Linewidth', 1); 
set(allaxes, 'FontName', 'Helvetica', 'LineWidth', 2, ...
    'FontSize', 18, 'box', 'on')
