%%
function [Answer_Train,Answer_Test,Answer2,CTime] = ModelAnn_New(Data)

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

%% Handle NaN values
Target_Train(isnan(Target_Train)) = mean(Target_Train,'omitnan');
Target_Test(isnan(Target_Test)) = mean(Target_Test,'omitnan');
for col = 1:size(Input_Train,2)
    nan_idx = isnan(Input_Train(:,col));
    if any(nan_idx)
        Input_Train(nan_idx,col) = mean(Input_Train(:,col),'omitnan');
    end
end
for col = 1:size(Input_Test,2)
    nan_idx = isnan(Input_Test(:,col));
    if any(nan_idx)
        Input_Test(nan_idx,col) = mean(Input_Test(:,col),'omitnan');
    end
end

%% Train a separate ANN for each WEKA cluster
tic
Cluster_IDs = unique(LM);
ann_nets = cell(1, max(Cluster_IDs));

param.hiddenLayerNum = 2;
param.TF = {'tansig','tansig','softmax'};
param.trainRatio = 0.70;
param.valRatio = 0.15;
param.testRatio = 0.15;
param.trainFcn = 'trainbr';
param.performParam.regularization = 0.2;
param.performFcn = 'crossentropy';
param.show = 1;
param.showWindow = 0;
param.showCommandLine = 0;
param.epochs = 2000;
param.goal = 0.0001;
param.max_fail = 200;

for c = Cluster_IDs
    idx_train = find(LM_Train == c);
    idx_test = find(LM_Test == c);

    if numel(idx_train) < 5 || numel(unique(Target_Train(idx_train))) < 2
        ann_nets{c} = [];
        continue;
    end

    n_cluster = numel(idx_train);
    if n_cluster < 20
        param.hiddenLayerSize = [3, 2];
    elseif n_cluster < 50
        param.hiddenLayerSize = [5, 3];
    else
        param.hiddenLayerSize = [10, 5];
    end

    Inp_tr = Input_Train(idx_train,:);
    Tar_tr = Target_Train(idx_train);
    if isempty(idx_test)
        Inp_te = Inp_tr(1,:);
        Tar_te = Tar_tr(1);
    else
        Inp_te = Input_Test(idx_test,:);
        Tar_te = Target_Test(idx_test);
    end

    try
        [~,~,~,~,net] = ML_ANN(Inp_tr, Inp_te, Tar_tr, Tar_te, param);
        ann_nets{c} = net;
    catch
        ann_nets{c} = [];
    end
end
CTime = toc;

%% Predict using cluster-specific ANN models
Score_Train = zeros(NTrain, 1);
for i = 1:NTrain
    c = LM_Train(i);
    if ~isempty(ann_nets{c})
        out = ann_nets{c}(Input_Train(i,:)');
        Score_Train(i) = out(1);
    else
        Score_Train(i) = 0.5;
    end
end

nTest = size(Input_Test,1);
Score_Test = zeros(nTest, 1);
for i = 1:nTest
    c = LM_Test(i);
    if ~isempty(ann_nets{c})
        out = ann_nets{c}(Input_Test(i,:)');
        Score_Test(i) = out(1);
    else
        Score_Test(i) = 0.5;
    end
end

Y_Train_ANN = double(Score_Train >= 0.5);
Y_test_ANN = double(Score_Test >= 0.5);

%% Evaluate Train
[confMat, ~] = confusionmat(Target_Train, Y_Train_ANN);
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
[confMat, ~] = confusionmat(Target_Test, Y_test_ANN);
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
save('ANN_results.mat')
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ANN ROC Curve (AUC = %.3f)', AUC_all));
grid on;
print('ROC_AUC_ANN','-dpng','-r300');
Answer2=[precision_train,OA_train,F1_Score_train,Recall_Train,MCC_Train,AUC_Train;precision_test,OA_test,F1_Score_test,Recall_Test,MCC_Test,AUC_Test];
close all

end
