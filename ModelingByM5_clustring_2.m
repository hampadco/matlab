%%
clear
clc
load('Cluster_index.mat')
%%
Data = table2array(readtable("date (2).csv"));
for i = 1: size(Data,1)
   L_modeled_score(i) = WekaM5TP_clustring(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6),idx(i));
end
%
L_modeled_score = L_modeled_score';
L_modeled = L_modeled_score;

nTrain = round(0.7*size(L_modeled_score,1));
%
% L_modeled (L_modeled_score>=0.5) = 1;
% L_modeled (L_modeled_score<0.5) = 0;
Target_Train = Data(1:nTrain,end);
Y_Train = L_modeled(1:nTrain,1);
Target_Test = Data(nTrain+1:end,end);
Y_test = L_modeled(nTrain+1:end,1);
Score_Train = L_modeled_score(1:nTrain,1);
Score_Test = L_modeled_score(nTrain+1:end,1);
%%
[confMat, order] = confusionmat(Target_Train, Y_Train);
TP = confMat(2,2);
FP = confMat(1,2);
TN = confMat(1,1);
FN = confMat(2,1);
precision_train = TP / (TP + FP);
OA_train = (TP+TN) / (TP + TN + FP + FN);
F1_Score_train = (2*TP) / (2*TP + FP + FN);
Recall_Train = TP / (TP + FN);
MCC_Train = (TP*TN - FN*FP) / sqrt((TP + FN)*(TN+FP)*(TP+FP)*(TN+FN));
[confMat, order] = confusionmat(Target_Test, Y_test);
TP = confMat(2,2);
FP = confMat(1,2);
TN = confMat(1,1);
FN = confMat(2,1);
precision_test = TP / (TP + FP);
OA_test = (TP+TN) / (TP + TN + FP + FN);
F1_Score_test = (2*TP) / (2*TP + FP + FN);
Recall_Test = TP / (TP + FN);
MCC_Test = (TP*TN - FN*FP) / sqrt((TP + FN)*(TN+FP)*(TP+FP)*(TN+FN));
%% Plot ROC
[X,Y,T,AUC_Train] = perfcurve(Target_Train, Score_Train, 1);  % '1' is the positive class
[X,Y,T,AUC_Test] = perfcurve(Target_Test, Score_Test, 1);  % '1' is the positive class
