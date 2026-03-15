function [Answer_Train,Answer_Test,Answer2,CTime] = ModelAnn(Data,Pre_Trained)
%%
Target=Data(:,end);
Input=[Data(:,1:end-1)];
%%
tic
precision_best=-inf;
count1=0;
if Pre_Trained==0
for i_1=[2,3,5,10,15,20]
     count1 = count1 + 1;
     count2=0;
    for i_2=[2,3,5,10,15,20]
        count2 = count2 + 1;
        param.hiddenLayerSize=[i_1,i_2];
        param.hiddenLayerNum = 2;
        param.TF={'tansig','tansig','softmax'};
        param.trainRatio=0.70;
        param.valRatio=0.15;
        param.testRatio=0.15;
        param.trainFcn='trainbr';
        param.performParam.regularization = 0.2;
        param.performFcn='crossentropy';
        param.show=1;
        param.showWindow=0;
        param.showCommandLine=0;
        param.epochs=2000;
        param.goal=0.0001;
        param.max_fail=200;
        %% create data
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
        [Y_Train_ANN,Y_test_ANN,Target_Train,Target_Test,net]=ML_ANN(Input_Train,Input_Test, Target_Train,Target_Test,param);
        Score_Train = Y_Train_ANN;
        Score_Test = Y_test_ANN;
        Y_Train_ANN(Y_Train_ANN>=0.5) = 1;
        Y_Train_ANN(Y_Train_ANN<0.5) = 0;
        Y_test_ANN(Y_test_ANN>=0.5) = 1;
        Y_test_ANN(Y_test_ANN<0.5) = 0; 
        [confMat, order] = confusionmat(Target_Test, Y_test_ANN);
        TP = confMat(2,2);
        FP = confMat(1,2);
        precision(count1,count2) = TP / (TP + FP);
        if precision(count1,count2)>precision_best
           precision_best=precision(count1,count2);
           i_best1=i_1;
           i_best2=i_2;
           net_best=net;
           Y_Train_ANN_best=Y_Train_ANN;
           Y_test_ANN_best=Y_test_ANN;
           Score_Train_best = Score_Train;
           Score_Test_best = Score_Test;
           Target_Train_best=Target_Train;
           Target_Test_best=Target_Test;
           % save('ANN_net','net_best')
        end
    end
CTime = toc;
disp(['Count',num2str(count1),' precision_best = ',num2str(precision_best)])
end
%% evaluate the model
[confMat, order] = confusionmat(Target_Train, Y_Train_ANN_best);
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
[confMat, order] = confusionmat(Target_Test, Y_test_ANN_best);
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
[X,Y,T,AUC_Train] = perfcurve(Target_Train, Score_Train_best, 1);  % '1' is the positive class
[X,Y,T,AUC_Test] = perfcurve(Target_Test, Score_Test_best, 1);  % '1' is the positive class
save('Tree_results.mat')
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--'); % reference line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.3f)', AUC_Test));
grid on;
print(['ROC_AUC_Tree'],'-dpng','-r300');  % saves as PNG
Answer2=[precision_train,OA_train,F1_Score_train,Recall_Train,MCC_Train,AUC_Train;precision_test,OA_test,F1_Score_test,Recall_Test,MCC_Test,AUC_Test];
else
load("ANN_net.mat")
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
Y_Train_ANN_best = net_best(Input_Train')';
Y_test_ANN_best = net_best(Input_Test')';
Score_Train = Y_Train_ANN_best;
Score_Test = Y_test_ANN_best;
Y_Train_ANN_best(Y_Train_ANN_best>=0.5) = 1;
Y_Train_ANN_best(Y_Train_ANN_best<0.5) = 0;
Y_test_ANN_best(Y_test_ANN_best>=0.5) = 1;
Y_test_ANN_best(Y_test_ANN_best<0.5) = 0; 
%% evaluate the model
[confMat, order] = confusionmat(Target_Train, Y_Train_ANN_best);
TP_Train = confMat(2,2);
FP_Train = confMat(1,2);
TN_Train = confMat(1,1);
FN_Train = confMat(2,1);
Answer_Train = [TP_Train,FP_Train;TN_Train,FN_Train];
precision_train = TP_Train / (TP_Train + FP_Train);
OA_train = (TP_Train+TN_Train) / (TP_Train + TN_Train + FP_Train + FN_Train);
F1_Score_train = (2*TP_Train) / (2*TP_Train + FP_Train + FN_Train);
Recall_Train = TP_Train / (TP_Train + FN_Train);
MCC_Train = (TP_Train*TN_Train - FN_Train*FP_Train) / sqrt((TP_Train + FN_Train)*(TN_Train+FP_Train)*(TP_Train+FP_Train)*(TN_Train+FN_Train));
[confMat, order] = confusionmat(Target_Test, Y_test_ANN_best);
TP_Test = confMat(2,2);
FP_Test = confMat(1,2);
TN_Test = confMat(1,1);
FN_Test = confMat(2,1);
Answer_Test = [TP_Test,FP_Test;TN_Test,FN_Test];
precision_test = TP_Test / (TP_Test + FP_Test);
OA_test = (TP_Test+TN_Test) / (TP_Test + TN_Test + FP_Test + FN_Test);
F1_Score_test = (2*TP_Test) / (2*TP_Test + FP_Test + FN_Test);
Recall_Test = TP_Test / (TP_Test + FN_Test);
MCC_Test = (TP_Test*TN_Test - FN_Test*FP_Test) / sqrt((TP_Test + FN_Test)*(TN_Test+FP_Test)*(TP_Test+FP_Test)*(TN_Test+FN_Test));
%% Plot ROC
CTime = inf;
[X1,Y1,T,AUC_Train] = perfcurve(Target_Train, Score_Train, 1);  % '1' is the positive class
[X2,Y2,T,AUC_Test] = perfcurve(Target_Test, Score_Test, 1);  % '1' is the positive class
[X,Y,T,AUC_all] = perfcurve([Target_Train;Target_Test], [Score_Train;Score_Test], 1);  % '1' is the positive class
save('ANN_results.mat')
figure;
plot(X, Y, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--'); % reference line
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ANN ROC Curve (AUC = %.3f)', AUC_all));
grid on;
print(['ROC_AUC_ANN'],'-dpng','-r300');  % saves as PNG
Answer2=[precision_train,OA_train,F1_Score_train,Recall_Train,MCC_Train,AUC_Train;precision_test,OA_test,F1_Score_test,Recall_Test,MCC_Test,AUC_Test];
close all

end
