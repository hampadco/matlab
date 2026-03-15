function [Answer_Train,Answer_Test,Answer2] = ModelAnn(Data)
%%
%%
Target=Data(:,end);
Input=[Data(:,1:end-1)];
%%
tic
precision_best=-inf;
count1=0;
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
%%
