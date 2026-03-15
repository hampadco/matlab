%%
%%
clear
clc
%%
load("ANN_net.mat")
Data = table2array(readtable("data2.csv"));
%%
Target=Data(:,end);
Input=[Data(:,1:end-1)];
%%
precision_best=-inf;
count1=0;
%%
NTrain=ceil(0.7*numel(Target));
Input_Train=Input(1:NTrain,:);
Input_Test=Input(NTrain+1:end,:);
Target_Train=Target(1:NTrain,1);
Target_Test=Target(NTrain+1:end,1);
%%
Target_Train(isnan(Target_Train)==1)=nanmean(Target_Train);
Target_Test(isnan(Target_Test)==1)=nanmean(Target_Test);
Input_Test(isnan(Input_Test(:,2))==1,2)=nanmean(Input_Test(:,2));
Input_Test(isnan(Input_Test(:,3))==1,3)=nanmean(Input_Test(:,3));
Input_Test(isnan(Input_Test(:,4))==1,4)=nanmean(Input_Test(:,4));
%% train and test
Y_Train_ANN = net_best(Input_Train')';
Y_test_ANN = net_best(Input_Test')';
Score_Train = Y_Train_ANN;
Score_Test = Y_test_ANN;
Y_Train_ANN(Y_Train_ANN>=0.5) = 1;
Y_Train_ANN(Y_Train_ANN<0.5) = 0;
Y_test_ANN(Y_test_ANN>=0.5) = 1;
Y_test_ANN(Y_test_ANN<0.5) = 0; 
[confMat, order] = confusionmat(Target_Test, Y_test_ANN);
TP = confMat(2,2);
FP = confMat(1,2);
[confMat, order] = confusionmat(Target_Train, Y_Train_ANN);
TP = confMat(2,2);
FP = confMat(1,2);
TN = confMat(1,1);
FN = confMat(2,1);
precision_train_ANN = TP / (TP + FP);
OA_train_ANN = (TP+TN) / (TP + TN + FP + FN);
F1_Score_train_ANN = (2*TP) / (2*TP + FP + FN);
Recall_Train = TP / (TP + FN);
MCC_Train = (TP*TN - FN*FP) / sqrt((TP + FN)*(TN+FP)*(TP+FP)*(TN+FN));
[confMat, order] = confusionmat(Target_Test, Y_test_ANN);
TP = confMat(2,2);
FP = confMat(1,2);
TN = confMat(1,1);
FN = confMat(2,1);
precision_test_ANN = TP / (TP + FP);
OA_test_ANN = (TP+TN) / (TP + TN + FP + FN);
F1_Score_test_ANN = (2*TP) / (2*TP + FP + FN);
Recall_Test = TP / (TP + FN);
MCC_Test = (TP*TN - FN*FP) / sqrt((TP + FN)*(TN+FP)*(TP+FP)*(TN+FN));
%% Plot ROC
[X,Y,T,AUC_Train] = perfcurve(Target_Train, Score_Train, 1);  % '1' is the positive class
[X,Y,T,AUC_Test] = perfcurve(Target_Test, Score_Test, 1);  % '1' is the positive class
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--'); % reference line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', AUC_Test));
grid on;
print(['ROC_AUC_ANN'],'-dpng','-r300');  % saves as PNG
