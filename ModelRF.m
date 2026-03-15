%%
function [Answer_Train,Answer_Test,Answer2,CTime] = ModelRF(Data)

%%
Target=Data(:,end);
Input=Data(:,1:end-1);

%% Assign WEKA clusters
for i = 1:size(Data,1)
    LM(i) = WekaM5TP_GP(Data(i,2), Data(i,3), Data(i,7), Data(i,1), Data(i,4), Data(i,5), Data(i,6));
end

%% Create data split
NTrain=ceil(0.7*numel(Target));
Input_Train=Input(1:NTrain,:);
Input_Test=Input(NTrain+1:end,:);
Target_Train=Target(1:NTrain,1);
Target_Test=Target(NTrain+1:end,1);
LM_Train = LM(1:NTrain);
LM_Test = LM(NTrain+1:end);

%% Train a separate RF for each WEKA cluster
tic
nTrees = 100;
Cluster_IDs = unique(LM);
rf_models = cell(1, max(Cluster_IDs));

for c = Cluster_IDs
    idx_train = find(LM_Train == c);
    if numel(idx_train) < 2 || numel(unique(Target_Train(idx_train))) < 2
        rf_models{c} = [];
        continue;
    end
    rf_models{c} = TreeBagger(nTrees, Input_Train(idx_train,:), Target_Train(idx_train), ...
        'Method', 'classification', 'MinLeafSize', 3);
end
CTime = toc;

%% Predict using cluster-specific RF models
Score_Train = zeros(NTrain, 1);
for i = 1:NTrain
    c = LM_Train(i);
    if ~isempty(rf_models{c})
        [~, sc] = predict(rf_models{c}, Input_Train(i,:));
        Score_Train(i) = sc(2);
    else
        Score_Train(i) = 0.5;
    end
end

nTest = size(Input_Test,1);
Score_Test = zeros(nTest, 1);
for i = 1:nTest
    c = LM_Test(i);
    if ~isempty(rf_models{c})
        [~, sc] = predict(rf_models{c}, Input_Test(i,:));
        Score_Test(i) = sc(2);
    else
        Score_Test(i) = 0.5;
    end
end

Y_Train = double(Score_Train >= 0.5);
Y_test = double(Score_Test >= 0.5);

%% Evaluate Train
[confMat, ~] = confusionmat(Target_Train, Y_Train);
TP_Train = confMat(2,2);
FP_Train = confMat(1,2);
TN_Train = confMat(1,1);
FN_Train = confMat(2,1);
Answer_Train = [TP_Train,FP_Train;FN_Train,TN_Train];
precision_train = TP_Train / (TP_Train + FP_Train);
OA_train = (TP_Train+TN_Train) / (TP_Train + TN_Train + FP_Train + FN_Train);
F1_Score_train = (2*TP_Train) / (2*TP_Train + FP_Train + FN_Train);
Recall_Train = TP_Train / (TP_Train + FN_Train);
MCC_Train = (TP_Train*TN_Train - FN_Train*FP_Train) / sqrt((TP_Train + FN_Train)*(TN_Train+FP_Train)*(TP_Train+FP_Train)*(TN_Train+FN_Train));

%% Evaluate Test
[confMat, ~] = confusionmat(Target_Test, Y_test);
TP_Test = confMat(2,2);
FP_Test = confMat(1,2);
TN_Test = confMat(1,1);
FN_Test = confMat(2,1);
Answer_Test = [TP_Test,FP_Test;FN_Test,TN_Test];
precision_test = TP_Test / (TP_Test + FP_Test);
OA_test = (TP_Test+TN_Test) / (TP_Test + TN_Test + FP_Test + FN_Test);
F1_Score_test = (2*TP_Test) / (2*TP_Test + FP_Test + FN_Test);
Recall_Test = TP_Test / (TP_Test + FN_Test);
MCC_Test = (TP_Test*TN_Test - FN_Test*FP_Test) / sqrt((TP_Test + FN_Test)*(TN_Test+FP_Test)*(TP_Test+FP_Test)*(TN_Test+FN_Test));

%% Plot ROC
[X1,Y1,~,AUC_Train] = perfcurve(Target_Train, Score_Train, 1);
[X2,Y2,~,AUC_Test] = perfcurve(Target_Test, Score_Test, 1);
[X,Y,~,AUC_all] = perfcurve([Target_Train;Target_Test], [Score_Train;Score_Test], 1);
save('RF_results.mat')
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Random Forest ROC Curve (AUC = %.3f)', AUC_all));
grid on;
print('ROC_AUC_RF','-dpng','-r300');
Answer2=[precision_train,OA_train,F1_Score_train,Recall_Train,MCC_Train,AUC_Train;precision_test,OA_test,F1_Score_test,Recall_Test,MCC_Test,AUC_Test];
close all
end
